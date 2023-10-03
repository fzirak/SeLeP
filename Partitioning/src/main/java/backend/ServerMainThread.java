package backend;

import Configuration.Config;
import Util.ConfigParams;
import Util.UtilFUnctions;
import backend.Controller.*;
import backend.Database.DatabaseRepo;
import backend.disk.NewTileManager;
import backend.disk.PartitionManager;
import backend.util.*;

import java.io.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

import org.eclipse.jetty.server.CustomRequestLog;
import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.handler.HandlerList;
import org.eclipse.jetty.server.handler.RequestLogHandler;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;

import org.apache.commons.cli.*;

public class ServerMainThread {
    public static BufferedWriter log;
    public static AffinityMatrix aMatrix;
    public static Server server;
    public static QueryWindow qWindow;
    public static PartitionManager partitionManager;
    public static NewTileManager tileManager;
    public static int overLoadCount = 0;
    
    public static int k = 10;
    public static int maxQWindowSize = 500;
    public static double maxPartitionLoad = 0.5;
    public static double weightResetThreshold = 0.1;
    public static double partitionFillPortion = 0.9;
    public static double initialEmptyPartitions = 0.05;
    public static int maxPartitionSize = 128;
    
    public static int initSeed = 3;
    public static HashMap<String,Double> individualMaxPartitionSize;
    public static int totalPartitionAccess = 0;
    public static int totalReceivedQueries = 0;

    public static String fileName = ".txt";

    public static ArrayList<LogLine> queries;

    public static void main(String[] args) throws Exception {
        Config conf = new Config(false, true);
        if(ConfigParams.is_navigation){
            ConfigParams.dbname = "sdss_1";
            ConfigParams.logicalBlockSize = 8;
            ConfigParams.db_comp_file = ConfigParams.baseDirectoryString + "prefetching/NaviDBComps.txt";
            ConfigParams.db_comps = new DBComponents();
            conf.setUpDBComps();
        }

        CommandLineParser parser = new DefaultParser();
        Options options = new Options();
        options.addOption("mqws", "maxQWinSize", true, "Max size for query window and the l_p threshold");
        options.addOption("wrv", "weightResetVal", true, "The number multiples to weights after each repartitioning");
        options.addOption("k", "k", true, "k in load calculation");
        options.addOption("ep", "empPar", true, "Percentage of empty partitions(from all partitions)");
        options.addOption("fp", "fillPortion", true, "Portion of the partitions that will be initially full");
        options.addOption("ml", "maxLoad", true, "Initial value for max load");
        options.addOption("mps", "maxPartitionSize", true, "Maximum size of partition in unit of blocks");
        options.addOption("rfn", "fileName", true, "file suffix");


        try {
            CommandLine cmd = parser.parse(options, args);
            if (cmd.hasOption("maxQWinSize"))
                maxQWindowSize = Integer.parseInt(cmd.getOptionValue("maxQWinSize"));
            if (cmd.hasOption("weightResetVal"))
                weightResetThreshold = Float.parseFloat(cmd.getOptionValue("weightResetVal"));
            if (cmd.hasOption("k"))
                k = Integer.parseInt(cmd.getOptionValue("k"));
            if (cmd.hasOption("empPar"))
                initialEmptyPartitions = Float.parseFloat(cmd.getOptionValue("empPar"));
            if (cmd.hasOption("fillPortion")) {
                System.out.println("received fillPortion");
                partitionFillPortion = Float.parseFloat(cmd.getOptionValue("fillPortion"));
            }
            if (cmd.hasOption("maxLoad"))
                maxPartitionLoad = Double.parseDouble(cmd.getOptionValue("maxLoad"));
            if (cmd.hasOption("maxPartitionSize"))
                maxPartitionSize = Integer.parseInt(cmd.getOptionValue("maxPartitionSize"));
            if (cmd.hasOption("fileName")){
                System.out.println("got the file name in params");
                fileName = cmd.getOptionValue("fileName") + ".txt";
                System.out.println(fileName);
            }
        } catch (ParseException | NumberFormatException e) {
            System.err.println("Error parsing command-line arguments: " + e.getMessage());
        }

        partitionManager = new PartitionManager();
        tileManager = new NewTileManager();
        initializePartitionAndTileManagers(); // comment this
        individualMaxPartitionSize = new HashMap<>();
        aMatrix = new AffinityMatrix();
        qWindow = new QueryWindow(maxQWindowSize);
        
        if (ConfigParams.dbname.equals("tpcds")){
            System.out.println("reading local queries");
            queries = UtilFUnctions.readQueriesFromFile("tpcds_trainB16.ser");
        }

        try {
            log = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("perflog.txt")));
        } catch (IOException e) {
            System.out.println("Couldn't open logfile");
            e.printStackTrace();
            return;
        }

        // start the server here
        server = new Server(ConfigParams.serverPort);
        ServletContextHandler context = new ServletContextHandler(ServletContextHandler.SESSIONS);
        context.setContextPath("/gettile");
        server.setHandler(context);
        context.addServlet(new ServletHolder(new LoginUser()), "/login");
        context.addServlet(new ServletHolder(new ExecuteQuery()), "/exec");
        context.addServlet(new ServletHolder(new TestQuery()), "/test");
        context.addServlet(new ServletHolder(new ServerStatus()), "/ready");
        context.addServlet(new ServletHolder(new ServerData()), "/serverdata");
        context.addServlet(new ServletHolder(new ServerControl()), "/servercntl");

        // uncomment these
        // aMatrix = UtilFUnctions.readAffMatrixFromFile("aMatrix" + fileName);
        // partitionManager = UtilFUnctions.readPartitionManagerFromFile("pManager" + fileName);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Shutting down server...");
            try {
                server.stop();
            } catch (Exception e) {
                e.printStackTrace();
            } 
            System.out.println("Server stopped");
        }));

        server.start();
    }

    public static void readQueryLocally(LogLine receivedQuery, String queryIdx){
        LogLine localQ = queries.get(Integer.valueOf(queryIdx));
        System.out.println("\nTotal " + localQ.query.result_set.size() + " tiles.");
        ServerMainThread.processNewQuery(localQ);
    }

    private static void initializePartitionAndTileManagers() {
        ArrayList<String> tables = new ArrayList<>();
        if(ConfigParams.is_navigation){
            tables.add("objnavi0");
            tables.add("objnavi1");
            tables.add("objnavi2");
            tables.add("objnavi3");
        }
        else
            tables = read_table_list();

        int initialCap = (int) Math.ceil(maxPartitionSize * partitionFillPortion);
        for (String table : tables) {
            System.out.println(table);
            ArrayList<String> result_set = new ArrayList<>();
            ArrayList<Integer> blockRange= DatabaseRepo.getTableBlocknumbers(table);
            for(int i : blockRange){
                result_set.add(table + "_" + i);
            }
            System.out.println(result_set.size());
            Partition newP = new Partition("p" + partitionManager.incrementIndex());
            for(String block : result_set){
                    // create the tile and insert it to partition
                NewTile tile = new NewTile(block);
                tile.setPartition_id(newP.partition_id);
                newP.addToTileList(tile.tid);
                tileManager.addTile(tile);
                if(newP.getSize() >= initialCap){
                    partitionManager.addPartitionWithLoad(newP, 0);
                    newP = new Partition("p" + partitionManager.incrementIndex());
                }
            }
            if(newP.getSize() > 0 && !partitionManager.containsPid(newP.partition_id)){
                partitionManager.addPartitionWithLoad(newP, 0);
            }
        }
        int additionalPartitions = (int) Math.ceil(initialEmptyPartitions * partitionManager.partitions.size());
        for(int i = 0; i < additionalPartitions; i++){
            // add some empty partitions in case:)
            partitionManager.addPartitionWithLoad(new Partition("p" + partitionManager.incrementIndex()), 0);
        }
        System.out.println("initial number of partitions: " + partitionManager.partitions.size());
    }

    public static ArrayList<String> read_table_list() {
        String file_name = (ConfigParams.is_navigation) ? "navi_tableLookUp.txt" : "tableLookUp.txt"; //tpcds_
        File file = new File(
                ConfigParams.baseDirectoryString + "pyprefetcher/Data/" + file_name);

        ArrayList<String> tables = new ArrayList<>();
        try {
            FileInputStream fis = new FileInputStream(file);
            Scanner sc = new Scanner(fis);
            String line;
            while (sc.hasNextLine()) {
                line = sc.nextLine().replaceAll("\\s", "").toLowerCase();
                tables.add(line);
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        return tables;
    }

    public static void updateAffinities(ArrayList<String> result_set) {
        System.out.println("updating affinities");
        for(String blockID : result_set){
            if(result_set.size() > ConfigParams.resSizeLimit)
                aMatrix.partialUpdateTileAccess(blockID, result_set, tileManager, maxQWindowSize);
            else
                aMatrix.updateTileAccess(blockID, result_set, maxQWindowSize);
        }
        System.out.println("update complete");
    }

    public static String printServerState(){
        System.out.println("got the print req");
        double sum = 0;
        double maxLoad = 0;
        double sumOfSquares = 0;
        int zeroLoadCount = 0;
        for (Double load : partitionManager.loads.values()) {
            sum += load;
            sumOfSquares += load * load;
            if (load > maxLoad)
                maxLoad = load;
            if (load == 0)
                zeroLoadCount ++;
        }

        double n = (double)partitionManager.partitions.size();
        double avgPAcc = (double) totalPartitionAccess/ (double) totalReceivedQueries;
        double avgL = sum/n;
        double variance = (sumOfSquares - (sum * sum) / n) / (n - 1);
        double standardDeviation = Math.sqrt(variance);

        StringBuilder sb = new StringBuilder();
        sb.append("\n------------- Server statistics -------------");
        sb.append("\nTotal loads: ").append(sum);
        sb.append("\nAvg load: ").append(avgL);
        sb.append("\nVariance load: ").append(variance);
        sb.append("\nStd load: ").append(standardDeviation);
        sb.append("\nMax load: ").append(maxLoad);
        sb.append("\nCount of overloads: ").append(overLoadCount);
        sb.append("\nNumber of partitions: ").append(partitionManager.partitions.size());
        sb.append("\nNumber of empty partitions: ").append(partitionManager.getNumEmptyPartitions());
        sb.append("\nNumber of partitions with zero load: ").append(zeroLoadCount);
        sb.append("\nTotal partition access: ").append(totalPartitionAccess);
        sb.append("\nAvg partition access: ").append(avgPAcc);

        System.out.println(sb.toString());
        return sb.toString();
    }

    public static void processNewQuery(LogLine q) {
        qWindow.insertQuery(q);
        updateAffinities(q.query.result_set);
        ArrayList<String> requested_partitions = getQueryPartitionsIBM(q);
        totalPartitionAccess += requested_partitions.size();
        totalReceivedQueries ++;
        System.out.println("requested partitions are: " + requested_partitions);
        partitionManager.updateLoads(requested_partitions, aMatrix, k);
        //update partitions based on the loads
//        if(partitionManager.detectOverload(maxPartitionLoad)){ // qWindow.getSize() >= maxQWindowSize ||
        if (qWindow.getSize() >= maxQWindowSize) {
            HashMap<String, Double> partitionLoad = partitionManager.getOverloadPartitions(maxPartitionLoad);
            if (partitionLoad.size() == 0){
                System.out.println("no overload detected after n query");
                System.out.println("new query is processed");
                aMatrix.multiplyWeights(weightResetThreshold);
                qWindow.clearQueryWindow();
                return;
            }
            for(Map.Entry<String, Double> pl : partitionLoad.entrySet()) {
                Partition mostOverload = partitionManager.partitions.get(pl.getKey());
                if (mostOverload.getPartitionLoad(aMatrix, k) < maxPartitionLoad) {
                    continue;
                }
                System.out.println("******* load before " + partitionManager.loads.get(pl.getKey()));
                System.out.println("******* size before " + mostOverload.getSize());
                updatePartitions(mostOverload);
                System.out.println("******* load after " + partitionManager.loads.get(pl.getKey()));
                System.out.println("******* size after " + mostOverload.getSize());
                if (partitionManager.loads.get(pl.getKey()) > maxPartitionLoad) {
                    maxPartitionLoad = 1.05 *partitionManager.loads.get(pl.getKey());
                    System.out.println("******* load after tar " + partitionManager.loads.get(pl.getKey()));
                }

                overLoadCount++;
            }
            aMatrix.multiplyWeights(weightResetThreshold);
            qWindow.clearQueryWindow();
        }
        System.out.println("new query is processed");

    }

    public static void addResPartition(LogLine q, String fileName) {
        ArrayList<String> requested_partitions = getQueryPartitionsIBM(q);
        writeQueryToFile(q, requested_partitions, fileName);

    }

    private static void writeQueryToFile(LogLine log, ArrayList<String> requested_partitions, String fileName) {
        System.out.println("writing to " + fileName + "WB" + ConfigParams.logicalBlockSize + "WP" + maxPartitionSize + ".txt");
        File fout = new File(fileName + "WB" + ConfigParams.logicalBlockSize + "WP" + maxPartitionSize + ".txt");
        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(fout, true);
            BufferedWriter fw = new BufferedWriter(new OutputStreamWriter(fos));
            String pattern = "MM/dd/yyyy hh:mm:ss a";
            DateFormat df = new SimpleDateFormat(pattern);
            String line = log.seqNUm + "||" + df.format(log.theTime) + "||" + log.clientIP + "||" +
                    log.resSize + "||" + log.query.statement + "||" + log.query.result_set + "||" + requested_partitions + "\n";
            fw.write(line);

            fw.close();
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void updatePartitions(Partition mostOverload) {
        int count = 0;
        int maxSizeLimit = getMaxClumpSize();
        System.out.println("in update partition");
        double mostOverLoadedParLoad = mostOverload.getPartitionLoad(aMatrix, k);
        System.out.printf("MaxParLoad=%f and found %s as most overloaded with load = ",
                maxPartitionLoad, mostOverload.partition_id);
        System.out.println(mostOverLoadedParLoad);
        ArrayList<String> m = new ArrayList<>();
        String d = "";
        Clump clump = new Clump();
        int lookAhead = 5;
        Set<String> affectedPartitions = new HashSet<>();
        while(mostOverload.getPartitionLoad(aMatrix, k) > maxPartitionLoad){
            count++;
            String clumpNeighbor = findClumpNeighbor(m, d);
            if(m.size() == 0){
                System.out.println("case 1-0");
                m.add(mostOverload.getHottestBlock(aMatrix));
                affectedPartitions.add(mostOverload.partition_id);
                System.out.printf("added %s to m[0]\n", m.get(0));
                d = findInitialCPartition(m);
                System.out.printf("d= %s\n", d);
                if(d.equals("")){
                    System.out.println("could not find a candidate partition!!");
//                    return;
                }
            }
            else if(!clumpNeighbor.equals("")){
                System.out.println("case 1-1");
                m.add(clumpNeighbor);
                affectedPartitions.add(tileManager.getTilePartition(clumpNeighbor));
                d = updateCPartition(m, d, mostOverload.partition_id);
                System.out.println("new d is " + d);
                if(m.size() == maxSizeLimit){
                    System.out.println("case 1-1-2");
                    doneReParWithNewPartition(m, clump, affectedPartitions, mostOverLoadedParLoad);
                    break;
                }
            }
            else{
                System.out.println("case 1-2");
                if(!clump.isEmpty()){
                    System.out.println("case 1-2-1");
                    moveClumpToDestPartition(clump);
                    affectedPartitions.add(clump.candidatePartition);
                    partitionManager.updateLoads(new ArrayList<>(affectedPartitions), aMatrix, k);
                    break;
                }
                else{
                    if(m.size() == maxSizeLimit){
                        System.out.println("case 1-2-2-2");
                        doneReParWithNewPartition(m, clump, affectedPartitions, mostOverLoadedParLoad);
                        break;
                    }
                    if (maxPartitionLoad < mostOverLoadedParLoad)
                        maxPartitionLoad = 1.05 *mostOverLoadedParLoad;
                    break;
                }
            }

            if(feasible(m, d)){
                System.out.println("case 2-1");
                clump.setTiles(m);
                clump.setCandidatePartition(d);
            } else if (!clump.isEmpty()) {
                System.out.println("case 2-2");
                System.out.println("reduced lookAhead");
                lookAhead--;
            }
            if (lookAhead == 0){
                System.out.println("lookAhead is zero");
                moveClumpToDestPartition(clump);
                affectedPartitions.add(clump.candidatePartition);
                partitionManager.updateLoads(new ArrayList<>(affectedPartitions), aMatrix, k);
                break;
            }
            System.out.println("end of update partition loop");
//            break;
        }
        System.out.println("Repartitioning is done");
    }

    private static int getMaxClumpSize() {
        String l = partitionManager.getLeastFilledPartition();
        return maxPartitionSize - partitionManager.partitions.get(l).getSize();
    }

    private static void doneReParWithNewPartition(
            ArrayList<String> m, Clump clump, Set<String> affectedPartitions, double maxLoad) {
        Partition newPartition = partitionManager.partitions.get(partitionManager.getLeastFilledPartition());
        System.out.println("calling doneReParWithNewPartition function with newPartition=" + newPartition.partition_id);
        if (maxPartitionSize - newPartition.getSize() < m.size()){
            System.out.println("partition does not have enough space, has " + (maxPartitionSize - newPartition.getSize()) + " needs" + m.size());
        }
        double totalSenderDelta = calculateTotalSenderDelta(affectedPartitions, m, k);
        double receiverDelta = receiverDelta(m, newPartition.partition_id, k);
        if(totalSenderDelta + receiverDelta >= 0){ // the repartitioning is not beneficial:)) rollback
            System.out.println("repartitioning rollback!! condition 1");
            if(maxLoad > maxPartitionLoad)
                maxPartitionLoad = 1.05 *maxLoad;
            return;
        }
        if (affectedPartitions.size() == 1 && newPartition.getSize() == 0 &&
                m.size() == partitionManager.partitions.get(new ArrayList<>(affectedPartitions).get(0)).getSize()){
            // this means that it will toggle between 2 empty partitions
            System.out.println("repartitioning rollback!! condition 2");
            if(maxLoad > maxPartitionLoad)
                maxPartitionLoad = 1.05 *maxLoad;
            return;
        }
        clump.setTiles(m);
        clump.setCandidatePartition(newPartition.partition_id);
        moveClumpToDestPartition(clump);
        affectedPartitions.add(newPartition.partition_id);
        partitionManager.updateLoads(new ArrayList<>(affectedPartitions), aMatrix, k);
        if (maxPartitionLoad < partitionManager.loads.get(newPartition.partition_id))
            maxPartitionLoad = 1.05 *partitionManager.loads.get(newPartition.partition_id);
    }

    private static double calculateTotalSenderDelta(Set<String> senders, ArrayList<String> m, int k) {
        double totalDelta = 0.0;
        for(String sender_pid : senders) {
            totalDelta += senderDelta(m, sender_pid, k);
        }
        return totalDelta;
    }

    private static void moveClumpToDestPartition(Clump clump) {
        System.out.println("removing " + clump.tiles.size() + " blocks");
        for(String tid : clump.tiles){
            String previousPartitionId = tileManager.getTilePartition(tid);
            if(previousPartitionId.equals(clump.candidatePartition)){
                System.out.println("same partition");
                continue;
            }
            tileManager.setTilePartition(tid, clump.candidatePartition);
            partitionManager.removeTileFromPartition(previousPartitionId, tid);
            partitionManager.addTileToPartition(clump.candidatePartition, tid);
        }
    }

    private static boolean feasible(ArrayList<String> m, String d) {
        int d_current_size =  partitionManager.partitions.get(d).getSize();
        if (d_current_size + m.size() > maxPartitionSize){
            System.out.println("size problem in feasible function");
            return false;
        }
        // k_problem
        double d_load = partitionManager.partitions.get(d).getPartitionLoad(aMatrix, k);
        double dr = receiverDelta(m, d);
        boolean r = (d_load + dr < maxPartitionLoad) || dr <= 0;
        if(r)
            System.out.printf("feasible for %s is true\n", d);
        else
            System.out.printf("feasible for %s is false\n", d);
        return r;
    }

    private static double receiverDelta(ArrayList<String> m, String d) {
        return receiverDelta(m, d, k);
    }

    public static double receiverDelta(ArrayList<String> m, String d, int k) {
        ArrayList<String> d_tiles = partitionManager.partitions.get(d).tiles;
        double cost = 0;
        for(String tid_v : m){
            if(d_tiles.contains(tid_v))
                continue;

            for(Map.Entry<String, Double> tfreq_u : aMatrix.getTileAffinities(tid_v).freqs.entrySet()){
                if(d_tiles.contains(tfreq_u.getKey())) {
                    cost -= (k * tfreq_u.getValue());
                }
                else if (!m.contains(tfreq_u.getKey())){
                    cost += (k * tfreq_u.getValue());
                }
            }
        }
        return cost;
    }

    private static double senderDelta(ArrayList<String> m, String p) {
        return senderDelta(m, p, k);
    }

    public static double senderDelta(ArrayList<String> m, String p, int k) {
        ArrayList<String> p_tiles = partitionManager.partitions.get(p).tiles;
        double cost = 0;
        for(String tid_v : m){
            if (!p_tiles.contains(tid_v))
                continue;
            for(Map.Entry<String, Double> tfreq_u : aMatrix.getTileAffinities(tid_v).freqs.entrySet()){
                if(!p_tiles.contains(tfreq_u.getKey())) {
                    cost -= (k * tfreq_u.getValue());
                }
                else if (!m.contains(tfreq_u.getKey())){
                    cost += (k * tfreq_u.getValue());
                }
            }
        }
        return cost;
    }

    public static String updateCPartition(ArrayList<String> m, String d, String p0) {
        if(!feasible(m, d)){
            String a = getMostCoAccessedPartition(m, d);
            if(!a.equals(d) && feasible(m, a))
                return a;
            String l = partitionManager.getLeastFilledPartition();
            if(!l.equals(d) && (receiverDelta(m, a) < receiverDelta(m, l)) && feasible(m, l))
                return l;
        }
        return d;
    }

    private static String getMostCoAccessedPartition(ArrayList<String> m, String d) {
        Map<String, Double> pfreq = new HashMap<>();
        for(String tid:m){
            for (Map.Entry<String, Double> tfreq: aMatrix.getTileAffinities(tid).freqs.entrySet()){
                String t_partition = tileManager.getTilePartition(tfreq.getKey());
                if(!t_partition.equals(d)){
                    if(pfreq.containsKey(t_partition))
                        pfreq.put(t_partition, pfreq.get(t_partition)+ tfreq.getValue());
                    else
                        pfreq.put(t_partition, tfreq.getValue());
                }
            }
        }
        Double maxFreq = 0.0;
        String resPartition = "";
        for (Map.Entry<String, Double> pf: pfreq.entrySet()){
            if (pf.getValue() > maxFreq){
                maxFreq = pf.getValue();
                resPartition = pf.getKey();
            }
        }
        return resPartition;
    }

    public static String findClumpNeighbor(ArrayList<String> m, String d) {
        double maxFreq = 0;
        String mostFrequentNeighborID = "";
        for(String tid:m){
            String tempID = aMatrix.getMostCoAccessedIDP(tid, tileManager, d, m);
            if(tempID.equals("")) {
                continue;
            }
            double tempFreq = aMatrix.getAffinity(tid, tempID) * aMatrix.getTileAccessCount(tid);
            if (tempFreq > maxFreq){
                maxFreq = tempFreq;
                mostFrequentNeighborID = tempID;
            }
        }
        System.out.println("finding clump neighbour. found " + mostFrequentNeighborID);
        return mostFrequentNeighborID;
    }

    public static String findInitialCPartition(ArrayList<String> clump) {
        return aMatrix.getMostCoAccessedParForTile(clump.get(0), tileManager);
    }

    public static ArrayList<String> getQueryPartitionsCN(LogLine q) {
        Partition newPartition = new Partition("p" + String.valueOf(partitionManager.incrementIndex()));
        Set<String> partitions = new HashSet<>();
        for(String tid:q.query.result_set){
            NewTile tile = tileManager.getTile(tid);
            if(tile == null){
                tile = new NewTile(tid, newPartition.partition_id);
                newPartition.addToTileList(tid);
                tileManager.addTile(tile);
                if(newPartition.getSize() >= maxPartitionSize){
                    partitionManager.addPartition(newPartition);
                    newPartition = new Partition("p" + partitionManager.incrementIndex());
                }
            }
            partitions.add(tile.partition_id);
        }
        if(newPartition.getSize() > 0 && !partitionManager.containsPid(newPartition.partition_id)){
            partitionManager.addPartition(newPartition);
        }
        return new ArrayList<>(partitions);
    }

    public static ArrayList<String> getQueryPartitionsIBM(LogLine q) {
        Map<String, Double> partitions_count = new HashMap<>();
        ArrayList<NewTile> unAllocatedTile = new ArrayList<>();
        for(String tid:q.query.result_set){
            NewTile tile = tileManager.getTile(tid);
            if(tile == null){
                System.out.println("tid " + tid + " does not exist");
                continue;
            }
            partitions_count.merge(tile.partition_id, 1.0, Double::sum);
        }
        partitions_count = UtilFUnctions.sortByValue(partitions_count, false);
        Iterator<Map.Entry<String, Double>> it = partitions_count.entrySet().iterator();
        int i = 0;
        while(unAllocatedTile.size() > 0){
            if(!it.hasNext())
                break;
            String pid = it.next().getKey();
            Partition p = partitionManager.partitions.get(pid);
            while (p.getSize() < maxPartitionSize){
                if(unAllocatedTile.size() == 0)
                    break;
                NewTile tile = unAllocatedTile.get(0);
                unAllocatedTile.remove(0);
                tile.setPartition_id(pid);
                p.addToTileList(tile.tid);
                tileManager.addTile(tile);
                partitions_count.merge(tile.partition_id, 1.0, Double::sum);
            }
        }
        if(unAllocatedTile.size() > 0){ //still need to allocate tiles to partitions
            Partition newPartition = new Partition("p" + String.valueOf(partitionManager.incrementIndex()));
            for(NewTile tile : unAllocatedTile) {
                tile.setPartition_id(newPartition.partition_id);
                newPartition.addToTileList(tile.tid);
                tileManager.addTile(tile);
                partitions_count.merge(tile.partition_id, 1.0, Double::sum);
                if (newPartition.getSize() >= maxPartitionSize) {
                    partitionManager.addPartition(newPartition);
                    newPartition = new Partition("p" + partitionManager.incrementIndex());
                }
            }
            if(newPartition.getSize() > 0 && !partitionManager.containsPid(newPartition.partition_id)){
                partitionManager.addPartition(newPartition);
            }
        }
        return new ArrayList<>(partitions_count.keySet());
    }

    public static boolean isReadyToTest() {
        return false;
    }

    public static void processNewTestQuery(LogLine q) {
    }

    public static void returnServerInfo(String name, String fileName_) {
        try {
            if (name.equals("serialize")){
                System.out.println("got serialize");
                UtilFUnctions.storeAffMatrixToFile(aMatrix, fileName_ + ".ser");
                UtilFUnctions.storePartitionManagerToFile(partitionManager, fileName_ + ".ser");
                return;
            }

            FileWriter writer;
            if (name.equals("statistics"))
                writer = new FileWriter(fileName_ + ".txt", true);
            else
                writer = new FileWriter(
                    ConfigParams.dbname +  fileName_ + "B" + ConfigParams.logicalBlockSize + "P" + maxPartitionSize + ".txt", false);
            switch (name) {
                case "partitions":
                    writer.write(getPartitionString());
                    break;
                case "affinityMatrix":
                    for (AffinityMatrix.AffinityEntry a : aMatrix.affinities.values()) {
                        String res = a.bid + "(" + a.accessCount + "):" + a.freqs + "\n";
                        writer.write(res);
                    }
                    break;
                case "navi_partitions":
                    writer.write(getPartitionString());
                    break;
                case "navi_affinityMatrix":
                    for (AffinityMatrix.AffinityEntry a : aMatrix.affinities.values()) {
                        String res = a.bid + "(" + a.accessCount + "):" + a.freqs + "\n";
                        writer.write(res);
                    }
                    break;
                case "statistics":
                    fileName = fileName_ + ".txt";
                    writer.write(printServerState());
                    break;
            }
            writer.close();
        }catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static String getPartitionString() {
        StringBuilder res = new StringBuilder();
        for(Partition p : partitionManager.partitions.values()){
            res.append(p.toString());
            res.append('\n');
        }
        return String.valueOf(res);
    }

    public static String getAMatrixString(){
        StringBuilder res = new StringBuilder();
        for(AffinityMatrix.AffinityEntry a : aMatrix.affinities.values()){
            res.append(a.bid).append("(").append(a.accessCount).append("):").append(a.freqs);
            res.append('\n');
        }
        return String.valueOf(res);
    }

    public static void stopServer(){
        try {
            System.out.println("Shutting down the server");
            UtilFUnctions.storeAffMatrixToFile(aMatrix, "aMatrix" + fileName);
            UtilFUnctions.storePartitionManagerToFile(partitionManager, "pManager" + fileName);
            server.stop();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static void clearQueryWindow(){
        qWindow.clearQueryWindow();
    }
}
