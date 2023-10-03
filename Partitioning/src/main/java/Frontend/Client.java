package Frontend;

import Configuration.Config;
import Util.ConfigParams;
import Util.UtilFUnctions;
import backend.Database.DatabaseRepo;
import backend.util.DBComponents;
import backend.util.DBObject;
import backend.util.DBObjectType;
import backend.util.LogLine;
import me.tongfei.progressbar.ProgressBar;

import com.google.gson.Gson;
import org.apache.commons.cli.*;
import org.javatuples.Pair;


import java.io.*;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static backend.ServerMainThread.read_table_list;

/*
    http://127.0.0.1:8080/gettile/serverdata?name=affinityMatrix
    http://127.0.0.1:8080/gettile/serverdata?name=partitions
    http://127.0.0.1:8080/gettile/serverdata?name=statistics
*/

public class Client {
    public static String request_base_str = "http://" + ConfigParams.serverIP + ":" + ConfigParams.serverPort + "/gettile/";
    public static boolean readFromFile = false;
    public static ArrayList<Integer> errorList = new ArrayList<>();
    public static int nn = 0;
    public static int batchSize = 10000;
    public static boolean isInitialReq = true;
    public static boolean doTestWorkloads = true;
    public static String resFileName = "res";
    public static int maxPartitionSize = 128;

    public static boolean getQueriesLocally = true;
    public static void main(String[] args) throws Exception {
//        ArrayList<LogLine> queries = UtilFUnctions.readQueriesFromFile("new_workloadWB.txt");
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
        options.addOption("rff", "readFromFile", true, "Read queries from file");
        options.addOption("bs", "batchSize", true, "batch size for storing queries to file");
        options.addOption("nn", "nn", true, "iter number for storing queries to file");
        options.addOption("iir", "isInitialReq", true, "Determines if the queries should change affect the partitions");
        options.addOption("dtw", "doTestWorkloads", true, "Determines if the test queries should be stored");
        options.addOption("rfn", "resFileName", true, "Result file for partitioning evaluation");
        options.addOption("mps", "maxPartitionSize", true, "Maximum size of partition in unit of blocks");

        try {
            CommandLine cmd = parser.parse(options, args);
            if (cmd.hasOption("readFromFile"))
                readFromFile = Boolean.parseBoolean(cmd.getOptionValue("readFromFile"));
            if (cmd.hasOption("batchSize"))
                batchSize = Integer.parseInt(cmd.getOptionValue("batchSize"));
            if (cmd.hasOption("nn"))
                nn = Integer.parseInt(cmd.getOptionValue("nn"));
            if (cmd.hasOption("isInitialReq")) {
                isInitialReq = Boolean.parseBoolean(cmd.getOptionValue("isInitialReq"));
                System.out.println(Boolean.parseBoolean(cmd.getOptionValue("isInitialReq")));
                System.out.println(cmd.getOptionValue("isInitialReq"));
            }
            if (cmd.hasOption("doTestWorkloads"))
                doTestWorkloads = Boolean.parseBoolean(cmd.getOptionValue("doTestWorkloads"));
            if (cmd.hasOption("resFileName"))
                resFileName = cmd.getOptionValue("resFileName");
            if (cmd.hasOption("maxPartitionSize"))
                maxPartitionSize = Integer.parseInt(cmd.getOptionValue("maxPartitionSize"));
        } catch (org.apache.commons.cli.ParseException | NumberFormatException e) {
            System.err.println("Error parsing command-line arguments: " + e.getMessage());
        }
        System.out.println("is initial req is:");
        System.out.println(isInitialReq);
        System.out.println("do test workloads is:");
        System.out.println(doTestWorkloads);
        System.out.println(resFileName);

        batch_execute_queries_main();
        // simple_execute_queries_main();
        // test_partitioning_config_main();
        // justGetServerComponents();
    }

    private static void batch_execute_queries_main() throws UnsupportedEncodingException, IOException {
        // int nn = 1;
        ArrayList<LogLine> queries;
        Gson gson = new Gson();
        if(readFromFile) {
            queries = UtilFUnctions.readQueriesFromFile("adapt_trainB8.ser");//test_2 is the new workload(test_queries2WB)
            System.out.println(queries.size());
        }
        else {
            System.out.println("Making the LogLines");
            queries = UtilFUnctions.getQueries(ConfigParams.baseDirectoryString + 
                // "workloads/try4/all_testExcluded1.csv", 
                "workloads/try4/adaptive/sdss_1_all_trainB8.csv",
                "\\|\\|"
            );
            for (LogLine query : queries) {
                if (Integer.parseInt(query.seqNUm) % 100 == 0) {
                    System.out.println(query.seqNUm);
                }
                ArrayList<String> updatedResList = new ArrayList<>();

                for (String element : query.query.result_set) {
                    String[] parts = UtilFUnctions.rsplit1(element, "_");
                    if (parts.length == 2) {
                        if (Objects.equals(parts[1], "null"))
                            continue;
                        int number = Integer.parseInt(parts[1]);
                        String updatedbid = parts[0] + "_" + (number);
                        updatedResList.add(updatedbid);
                    }
                    else
                        System.out.println("error " + query.seqNUm);
                }
                query.query.result_set = updatedResList;
            }
           UtilFUnctions.storeQueriesToFile(queries, "adapt_trainB8.ser"); // all_testExcluded4.txt
        }

        int n = queries.size();
    //    int n = 520;
        if (isInitialReq) { 
            long res;
            LogLine query;
            for(int i=0; i<n; i++) {
                query = queries.get(i);
                String json = gson.toJson(query);
                res = sendRequest(request_base_str + "exec?alter=1", "query=" + URLEncoder.encode(json, "utf-8"), i);
                if (res < 0) {
                    System.out.println("error while executing query" + query.query.statement);
                }
            }
            sendPlainGetRequest(request_base_str + "serverdata?name=serialize&filename=sdss_1_adapt_partitions");
            sendPlainGetRequest(request_base_str + "serverdata?name=serialize&filename=sdss_1_adapt_affinityMatrix");
        }
        else{
            int batches = (int) Math.ceil((float) queries.size()/(float) batchSize);
            for(int j = nn; j < Math.min(nn+1, batches); j++){
                for(int i=j*batchSize; i<(Math.min(queries.size(), (j+1)*batchSize)); i++) {
                    LogLine query = queries.get(i);
                    String json = gson.toJson(query);
                    long res = sendRequest(request_base_str + "exec?alter=0&fname=sdss_1_adapt_all_train", "query=" + URLEncoder.encode(json, "utf-8"), i);
                    if (res < 0) 
                        System.out.println("error while executing query" + query.query.statement);
                }
            }
            System.out.println(errorList);
        }

        if(doTestWorkloads){
            System.out.println("Clearing the query window on the server");
            sendPlainGetRequest(request_base_str + "servercntl?command=clearwindow");
            for(int j = 0; j < 16; j++){
                sendPlainGetRequest(request_base_str + "serverdata?name=partitions&filename=_adapt_partitions" + j);
                sendPlainGetRequest(request_base_str + "serverdata?name=affinityMatrix&filename=_adapt_affinityMatrix" + j);
                queries = UtilFUnctions.getQueries(ConfigParams.baseDirectoryString + 
                    "workloads/try4/adaptive/sdss_1_adapt_test" + j + ".csv", "\\|\\|");
                for (LogLine query : queries) {
                    if (Integer.parseInt(query.seqNUm) % 100 == 0) {
                        System.out.println(query.seqNUm);
                    }
                    ArrayList<String> updatedResList = new ArrayList<>();

                    for (String element : query.query.result_set) {
                        String[] parts = UtilFUnctions.rsplit1(element, "_");
                        if (parts.length == 2) {
                            if (Objects.equals(parts[1], "null"))
                                continue;
                            int number = Integer.parseInt(parts[1]);
                            String updatedbid = parts[0] + "_" + (number);
                            updatedResList.add(updatedbid);
                        }
                        else
                            System.out.println("error " + query.seqNUm);
                    }
                    query.query.result_set = updatedResList;
                }
               
                long res;
                LogLine query;

                for(int ii=0; ii<queries.size(); ii++) {
                    query = queries.get(ii);
                    String json = gson.toJson(query);
                    res = sendRequest(request_base_str + "exec?alter=0&fname=sdss_1_adapt_test" + j, "query=" + URLEncoder.encode(json, "utf-8"), ii);
                    if (res < 0) 
                        System.out.println("error while executing query" + query.query.statement);
                }

                for(int ii=0; ii<queries.size(); ii++) {
                    query = queries.get(ii);
                    String json = gson.toJson(query);
                    res = sendRequest(request_base_str + "exec?alter=1", "query=" + URLEncoder.encode(json, "utf-8"), ii);
                    if (res < 0) {
                        System.out.println("error while executing query" + query.query.statement);
                    }
                }
            }           
            System.out.println(errorList);
        }        
    }


    public static void justGetServerComponents(){
        System.out.println("Getting the Partition Manger with\n\t" + request_base_str + "serverdata?name=partitions&filename=pManager" + resFileName);
        sendPlainGetRequest(request_base_str + "serverdata?name=partitions&filename=pManager" + resFileName); // resfilename is without partition_eval, it is just the suffix
        System.out.println("Getting the Affinity Matirix");
        sendPlainGetRequest(request_base_str + "serverdata?name=affinityMatrix&filename=affMat" + resFileName);
        System.out.println("shutting down the server");
        sendPlainGetRequest(request_base_str + "servercntl?command=stop");
    }

    public static ArrayList<LogLine> partition_initializer_workload() throws ParseException {
        ArrayList<String> tables = read_table_list();
        ArrayList<LogLine> queries = new ArrayList<>();
        DateFormat dateFormat = new SimpleDateFormat("MM/dd/yyyy hh:mm:ss a");
        Date date = dateFormat.parse("06/30/2012 12:58:17 am");
        String clientip = "180.76.5.91";
        for (String table : tables) {
            System.out.println(table);
            String sql = "select * from " + table;
            ArrayList<String> result_set = new ArrayList<>();
            Pair<Integer, Integer> blockRange= DatabaseRepo.getTableBlockRange(table);
            System.out.println(blockRange);
            for(int i=blockRange.getValue0(); i <= blockRange.getValue1(); i++){
                result_set.add(table + "_" + i);
            }
            System.out.println(result_set.size());
            LogLine new_log = new LogLine(date, clientip, 1, sql, result_set);
            queries.add(new_log);
            break;
        }
        return queries;
    }


    private static void test_partitioning_config_main() throws UnsupportedEncodingException, IOException {
        ArrayList<LogLine> queries;
        Gson gson = new Gson();
        if(readFromFile) {
            queries = UtilFUnctions.readQueriesFromFile("all_testExcluded4.txt");//test_2 is the new workload(test_queries2WB)
            System.out.println(queries.size());
        }
        else {
            System.out.println("reading the queries");
            queries = UtilFUnctions.getQueries(ConfigParams.baseDirectoryString + "workloads/try4/all_testExcluded.csv");
            for (LogLine query : queries) {
                if (Integer.parseInt(query.seqNUm) % 10 == 0) {
                    System.out.println(query.seqNUm);
                }
                ArrayList<String> updatedResList = new ArrayList<>();

                for (String element : query.query.result_set) {
                    String[] parts = element.split("_");
                    if (parts.length == 2) {
                        if (Objects.equals(parts[1], "null"))
                            continue;
                        int number = Integer.parseInt(parts[1]);
                        String updatedbid = parts[0] + "_" + (number / ConfigParams.logicalBlockSize);
                        if(updatedbid.equals("speclineall_31")){
                            System.out.println(query.query.toString());
                        }
                        if(!updatedResList.contains(updatedbid))
                            updatedResList.add(updatedbid);
                    }
                    else
                        System.out.println("error " + query.seqNUm);
                }
                query.query.result_set = updatedResList;
            }
           UtilFUnctions.storeQueriesToFile(queries, "all_testExcluded4.txt");
        }

        int n = queries.size();
    //    int n = 520;
        if (isInitialReq) { 
            long startSending = System.currentTimeMillis();   
            for(int i=0; i<n; i++) {
                LogLine query = queries.get(i);
                String json = gson.toJson(query);
                long res = sendRequest(request_base_str + "exec?alter=1", "query=" + URLEncoder.encode(json, "utf-8"), i);
                if (res < 0) {
                    System.out.println("error while executing query" + query.query.statement);
                }
            }
            long doneSending = System.currentTimeMillis();
            FileWriter writer = new FileWriter("partition_eval" + resFileName + ".txt", true);
            writer.write("Initial Partitioning time = " + (doneSending - startSending));
            writer.close();
            // read the sample file here and send it to the server to be stored in file
            sendPlainGetRequest(request_base_str + "serverdata?name=statistics&filename=partition_eval" + resFileName);
        }
        else{
            int batches = (int) Math.ceil((float) queries.size()/(float) batchSize);
            for(int j = nn; j < Math.min(nn+1, batches); j++){
                for(int i=j*batchSize; i<(Math.min(queries.size(), (j+1)*batchSize)); i++) {
                    LogLine query = queries.get(i);
                    String json = gson.toJson(query);
                    long res = sendRequest(request_base_str + "exec?alter=0&fname=sdss1_all_train", "query=" + URLEncoder.encode(json, "utf-8"), i);
                    if (res < 0) 
                        System.out.println("error while executing query" + query.query.statement);
                }
            }
            System.out.println(errorList);
        }

        justGetServerComponents();


        if (doTestWorkloads){
            for(int j = 1; j < 4; j++) {
                for (int jj = 1; jj < 3; jj++) {
                    String read_file_name = "sdss_1_test" + j + "_" + jj;
                    String write_file_name = read_file_name;

                    queries = UtilFUnctions.getQueries(
                            ConfigParams.baseDirectoryString + "workloads/try4/" + read_file_name + ".csv");
                    sendAllNoalterQueries(queries, gson, write_file_name + resFileName, queries.size());
                }
            }
            String[] files = {"test3_1t", "test3_1timely", "test3_2t", "test3_2timely", "test1_1gen", "test1_1gen"};
            for(int j=0; j < files.length/2; j++){
                String read_file_name = ConfigParams.baseDirectoryString + "workloads/try4/sdss_1_" + files[2*j] + ".csv";
                String write_file_name = files[2*j+1];

                queries = UtilFUnctions.getQueries(read_file_name);
                sendAllNoalterQueries(queries, gson, "sdss_1_" + write_file_name + resFileName, queries.size());
            }
        }
    }

    private static void simple_execute_queries_main() throws UnsupportedEncodingException, IOException {
        // int nn = 1;
        ArrayList<LogLine> queries;
        Gson gson = new Gson();
        if(readFromFile) {
            queries = UtilFUnctions.readQueriesFromFile("tpcds_trainB16.ser");//test_2 is the new workload(test_queries2WB)
            System.out.println(queries.size());
        }
        else {
            System.out.println("Making the LogLines");
            queries = UtilFUnctions.getQueries(ConfigParams.baseDirectoryString + 
                // "workloads/try4/all_testExcluded1.csv", 
                "workloads/tpcds/trainB1.csv",
                "\\$"
            );
            for (LogLine query : queries) {
                if (Integer.parseInt(query.seqNUm) % 100 == 0) {
                    System.out.println(query.seqNUm);
                }
                ArrayList<String> updatedResList = new ArrayList<>();

                for (String element : query.query.result_set) {
                    String[] parts = UtilFUnctions.rsplit1(element, "_");
                    if (parts.length == 2) {
                        if (Objects.equals(parts[1], "null"))
                            continue;
                        int number = Integer.parseInt(parts[1]);
                        String updatedbid = parts[0] + "_" + (number / ConfigParams.logicalBlockSize);
                        if(updatedbid.equals("speclineall_31")){
                            System.out.println(query.query.toString());
                        }
                        if(!updatedResList.contains(updatedbid))
                            updatedResList.add(updatedbid);
                    }
                    else
                        System.out.println("error " + query.seqNUm);
                }
                query.query.result_set = updatedResList;
            }
           UtilFUnctions.storeQueriesToFile(queries, "tpcds_train.ser"); // all_testExcluded4.txt
        }

        int n = queries.size();
    //    int n = 520;
        if (isInitialReq) { 
            long res;
            LogLine query;
            for(int i=0; i<n; i++) {
                query = queries.get(i);
                if (!getQueriesLocally){
                    String json = gson.toJson(query);
                    res = sendRequest(request_base_str + "exec?alter=1", "query=" + URLEncoder.encode(json, "utf-8"), i);
                }
                else{
                    String json = gson.toJson(new LogLine(query.theTime, query.clientIP, Integer.valueOf(query.seqNUm), query.query.statement, null));
                    res = sendRequest(request_base_str + "exec?alter=1&queryidx=" + String.valueOf(i), "query=" + URLEncoder.encode(json, "utf-8"), i);
                }
                if (res < 0) {
                    System.out.println("error while executing query" + query.query.statement);
                }
            }
        }
        else{
            int batches = (int) Math.ceil((float) queries.size()/(float) batchSize);
            for(int j = nn; j < Math.min(nn+1, batches); j++){
                for(int i=j*batchSize; i<(Math.min(queries.size(), (j+1)*batchSize)); i++) {
                    LogLine query = queries.get(i);
                    String json = gson.toJson(query);
                    long res = sendRequest(request_base_str + "exec?alter=0&fname=sdss_1_all_train", "query=" + URLEncoder.encode(json, "utf-8"), i);
                    if (res < 0) 
                        System.out.println("error while executing query" + query.query.statement);
                }
            }
            System.out.println(errorList);
        }
        
        sendPlainGetRequest(request_base_str + "serverdata?name=partitions&filename=tpcds_pManager"); // resfilename is without partition_eval, it is just the suffix
        sendPlainGetRequest(request_base_str + "serverdata?name=affinityMatrix&filename=tpcds_affMat");

        if (doTestWorkloads){
            for(int j = 1; j < 4; j++) {
                for (int jj = 1; jj < 3; jj++) {
                    // String read_file_name = ConfigParams.baseDirectoryString + "workloads/try4/sdss_1_test3_1t.csv";
                    // String write_file_name = "sdss_1_test3_1timely";
                    String read_file_name = "sdss_1_test" + j + "_" + jj;
                    String write_file_name = read_file_name;

                    queries = UtilFUnctions.getQueries(
                            ConfigParams.baseDirectoryString + "workloads/try4/" + read_file_name + ".csv");
                    sendAllNoalterQueries(queries, gson, write_file_name, queries.size());

                }
            }
            String[] files = {"test3_1t", "test3_1timely", "test3_2t", "test3_2timely", "test1_1gen", "test1_1gen"};
            // String[] files = {"test3_1", "test3_1"};
            for(int j=0; j < files.length/2; j++){
                String read_file_name = ConfigParams.baseDirectoryString + "workloads/try4/sdss_1_" + files[2*j] + ".csv";
                String write_file_name = files[2*j+1];

                queries = UtilFUnctions.getQueries(read_file_name);
                sendAllNoalterQueries(queries, gson, "sdss_1_" + write_file_name, queries.size());
            }
        }
    
    }

    private static void sendAllNoalterQueries(ArrayList<LogLine> queries, Gson gson, String fileName, int n) throws UnsupportedEncodingException {
        for (LogLine query : queries) {
            ArrayList<String> updatedResList = new ArrayList<>();
            for (String element : query.query.result_set) {
                String[] parts = UtilFUnctions.rsplit1(element, "_");
                if (parts.length == 2) {
                    if (Objects.equals(parts[1], "null"))
                        continue;
                    int number = Integer.parseInt(parts[1]);
                    String updatedbid;
                    if(fileName.equals("sdss_1_test3_1") || fileName.equals("sdss_1_test3_2") || fileName.equals("sdss_1_test1_1gen"))
                        updatedbid = parts[0] + "_" + number;
                    else
                        updatedbid = parts[0] + "_" + (number / ConfigParams.logicalBlockSize);
                    if (!updatedResList.contains(updatedbid))
                        updatedResList.add(updatedbid);
                } else
                    System.out.println("error " + query.seqNUm);
            }
            query.query.result_set = updatedResList;
        }

        for (int i = 0; i < n; i++) {
            LogLine query = queries.get(i);
            String json = gson.toJson(query);
            long res = sendRequest(request_base_str + "exec?alter=0&fname=" + fileName,
                    "query=" + URLEncoder.encode(json, "utf-8"), i);
            if (res < 0) {
                System.out.println("error while executing query" + query.query.statement);
            }
        }
    }

    public static long sendPlainGetRequest(String urlstring) {
        URL geturl = null;
        HttpURLConnection connection = null;
        BufferedReader reader = null;
        try {
            geturl = new URL(urlstring);
            if(geturl == null) {
                System.out.println("error in getting the url");
                return -1;
            }
            connection = (HttpURLConnection) geturl.openConnection();
            connection.setRequestMethod("GET");
            connection.setDoOutput(true);

            String output;
            reader = new BufferedReader(new InputStreamReader((connection.getInputStream())));
            while ((output = reader.readLine()) != null) {
                System.out.println(output);
            }

        } catch (MalformedURLException e) {
            System.out.println("error occurred while retrieving url object for: '"+urlstring+"'");
        } catch (IOException e) {
            System.out.println("Error retrieving response" );
            return -1;
//            e.printStackTrace();
        }

        if(connection != null) {
            connection.disconnect();
        }

        if(reader != null) {
            try {
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return 1;
    }

    public static long sendRequest(String urlstring, String queryParam, int i) {
//        System.out.println("sending request " + i);
        URL geturl = null;
        HttpURLConnection connection = null;
        BufferedReader reader = null;
        try {
            geturl = new URL(urlstring);
            if(geturl == null) {
                System.out.println("error in getting the url");
                errorList.add(i);
                return -1;
            }
//            System.out.println("sending request: " + geturl);
            connection = (HttpURLConnection) geturl.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            OutputStream os = connection.getOutputStream();
            os.write(queryParam.getBytes());
            os.flush();
            os.close();

            System.out.printf("%dth request response: ", i);
            String output;
            reader = new BufferedReader(new InputStreamReader((connection.getInputStream())));
            while ((output = reader.readLine()) != null) {
                System.out.println(output);
            }

        } catch (MalformedURLException e) {
            System.out.println("error occurred while retrieving url object for: '"+urlstring+"'");
            e.printStackTrace();
            errorList.add(i);
        } catch (IOException e) {
            System.out.println("Error retrieving response for request number " + i );
            errorList.add(i);
            e.printStackTrace();
            return -1;
        }

        if(connection != null) {
            connection.disconnect();
        }

        if(reader != null) {
            try {
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return 1;
    }

    private static void analyseQueriesResult(ArrayList<LogLine> queries) {
        System.out.println(queries.size());
        int sum = 0;
        Map<Integer, Integer> resfreq = new HashMap<>();
        for(LogLine query: queries){
            if(resfreq.containsKey(query.query.result_set.size())){
                int freq = resfreq.get(query.query.result_set.size()) + 1;
                resfreq.put(query.query.result_set.size(), freq);
            }
            else
                resfreq.put(query.query.result_set.size(), 1);
            sum += query.query.result_set.size();
        }
        System.out.println(sum/ queries.size());
        StringBuilder outline1 = new StringBuilder();
        StringBuilder outline2 = new StringBuilder();
        for(Map.Entry<Integer, Integer> ent : resfreq.entrySet()){
            outline1.append(ent.getKey()).append(", ");
            outline2.append(ent.getValue()).append(", ");
        }
        System.out.println(outline1);
        System.out.println(outline2);
        System.out.println("max = " + Collections.max(resfreq.keySet()));
    }
}
