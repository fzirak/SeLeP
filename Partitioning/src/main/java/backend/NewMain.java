package backend;

import Configuration.Config;
import Util.ConfigParams;
import Util.UtilFUnctions;
import backend.Database.DatabaseRepo;
import backend.util.*;
import org.javatuples.Pair;

import java.io.*;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static Util.ConfigParams.query_file_path;

import me.tongfei.progressbar.*;

class ParallelTask implements Runnable {
    private ArrayList<LogLine> queries;
    private int start;
    private int end;
    private int q_res_size_limit;

    public ParallelTask(ArrayList<LogLine> queries, int start, int end, int qrsl) {
        this.start = start;
        this.end = end;
        this.queries = queries;
        this.q_res_size_limit = qrsl;
    }

    @Override
    public void run() {
        try (ProgressBar pb = new ProgressBar("Processing batch " + start + ":", end - start)) {
            for (int i_ = start; i_ < end; i_++) {
                pb.step();
                LogLine query;
                synchronized (queries) {
                    query = queries.get(i_);
                }
                NewMain.executeQuery(query);

                if (query.query.result_set.size() < q_res_size_limit )
                    continue;
                else if (query.query.including_components.size() > 1){
                    int lim = 200;
                    while (query.query.result_set.size() > q_res_size_limit && lim > 10) {
                        String stmt = query.query.statement;
                        Pattern p = Pattern.compile("\\blimit\\s+\\d+\\b", Pattern.CASE_INSENSITIVE);
                        Matcher matcher = p.matcher(stmt);

                        if (matcher.find()) {
                            stmt = matcher.replaceAll("limit " + String.valueOf(lim));
                        } else {
                            stmt += " limit " + String.valueOf(lim);
                        }
                        query.query.statement = stmt;
                        NewMain.executeQuery(query);
                        lim *= 0.5;
                    }
                    // updateQueryResSize(query);
                    synchronized (queries) {
                        queries.set(i_, query);
                    }
                }
                else if(query.query.result_set.size() > q_res_size_limit * 1.25 ){
                    int lim = 400;
                    String stmt = query.query.statement;
                    while (query.query.result_set.size() > q_res_size_limit * 1.25 && lim > 10) {
                        Pattern p = Pattern.compile("\\blimit\\s+\\d+\\b", Pattern.CASE_INSENSITIVE);
                        Matcher matcher = p.matcher(stmt);

                        if (matcher.find()) {
                            stmt = matcher.replaceAll("limit " + String.valueOf(lim));
                        } else {
                            stmt += " limit " + String.valueOf(lim);
                        }
                        query.query.statement = stmt;
                        NewMain.executeQuery(query);
                        lim *= 0.5;
                    }
                    // updateQueryResSize(query);
                    synchronized (queries) {
                        queries.set(i_, query);
                    }
                }
            }
    
        }
    }
}


public class NewMain {
    public static BufferedWriter log;
    public static AffinityMatrix aMatrix;

    public static void main(String[] args) throws IOException {
        /*
        * use this to find out blocks each query accessed:
        * 1- readfromfile = false
        * 2- change queries file_path
        * 3- change fout file name
        * 4- if your query file contains seqNum in the beginning you should modify line 175 of getQueries function
        */
        boolean readFromFile = false;
        Config conf = new Config(false, true);
        if(ConfigParams.is_navigation){
            ConfigParams.dbname = "sdss_1";
            ConfigParams.logicalBlockSize = 8;
            ConfigParams.db_comp_file = ConfigParams.baseDirectoryString + "prefetching/NaviDBComps.txt";
            ConfigParams.db_comps = new DBComponents();
            conf.setUpDBComps();
        }
        String loadFilePath = "test_queries2WB.txt";
        String writeWBFilepath = ConfigParams.baseDirectoryString + "workloads/try4/"
                + ConfigParams.dbname + "_test1_1gen.csv";
                // + ConfigParams.dbname + "sdss_1_allB" + ConfigParams.logicalBlockSize + ".csv";
        String readQueriesFP =
               ConfigParams.baseDirectoryString + "workloads/try4/generated_queries_single_reg.txt";
        int batch_size = 800;

        try {
            log = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("perflog.txt")));
        } catch (IOException e) {
            System.out.println("Couldn't open logfile");
            e.printStackTrace();
            return;
        }

        ArrayList<LogLine> queries = new ArrayList<>();
        if(readFromFile) {
            queries = UtilFUnctions.readQueriesFromFile(loadFilePath);
            System.out.println(queries.size());

            File fout = new File(writeWBFilepath);
            FileOutputStream fos = new FileOutputStream(fout);

            BufferedWriter fw = new BufferedWriter(new OutputStreamWriter(fos));
            String pattern = "MM/dd/yyyy hh:mm:ss a";
            DateFormat df = new SimpleDateFormat(pattern);
            fw.write("seqNum||theTime||clientIP||row||statement||resultBlock");
            for(LogLine log: queries){
                if(log.query.result_set.size() == 0)
                    continue;
                String line = "\n" + log.seqNUm + "||" + df.format(log.theTime) + "||" + log.clientIP + "||" +
                        log.resSize + "||" + log.query.statement + "||" + log.query.result_set;
//            System.out.println(line);
                fw.write(line);
            }
            fw.close();
        }
        else {
            queries = UtilFUnctions.getQueries(readQueriesFP);
            String pattern = "MM/dd/yyyy hh:mm:ss a";
            DateFormat df = new SimpleDateFormat(pattern);
            int cycles = queries.size()/batch_size + ((queries.size() % batch_size != 0) ? 1 : 0);
            int start_idx = 0;
            int q_res_size_limit = 240;

            int numCores = 40;
            int batchSize = (int) Math.ceil((float)batch_size/(float)numCores);
        
            
            List<Thread> threads = new ArrayList<>();

            try (ProgressBar pbm = new ProgressBar("Processing queries:", cycles - start_idx)) {
                for (int i=start_idx; i< cycles; i++){
                    pbm.step();

                    for (int ii = 0; ii < numCores; ii++) {
                        int start = (ii * batchSize) + (i*batch_size);
                        int end = Math.min(queries.size(), Math.min((ii + 1) * batchSize, batch_size) + (i*batch_size));
                        
                        Runnable task = new ParallelTask(queries, start, end, q_res_size_limit); 
                        
                        Thread thread = new Thread(task);
                        threads.add(thread);
                        thread.start();
                    }
                    
                    for (Thread thread : threads) {
                        try {
                            thread.join();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }

                    File fout = new File(writeWBFilepath);
                    boolean append = i != 0;
                    FileOutputStream fos = new FileOutputStream(fout, append);
                    BufferedWriter fw = new BufferedWriter(new OutputStreamWriter(fos));

                    if(i == 0)
                        fw.write("seqNum||theTime||clientIP||row||statement||resultBlock");

                    for(int i_=i*batch_size; i_ < Math.min(queries.size(), (i+1)*batch_size); i_++){
                        LogLine log = queries.get(i_);
                        if(log.query.result_set.size() == 0)
                            continue;
                        String line = "\n" + log.seqNUm + "||" + df.format(log.theTime) + "||" + log.clientIP + "||" +
                                log.resSize + "||" + log.query.statement + "||" + log.query.result_set;
                        fw.write(line);
                        queries.set(i_, new LogLine());
                    }
                    fw.close();

                }

            }

        }

    }

    public static void updateQueryResSize(LogLine logline) {
        String query = logline.query.statement;
        try {
            Connection conn = DatabaseRepo.getDefaultPostgresqlConnection();
            PreparedStatement ps = conn.prepareStatement(query);
            ResultSet rs = ps.executeQuery();
            if (!rs.isBeforeFirst() ) {
                if(logline.resSize != 0)
                    logline.setResSize(0);
            }

            int resCount = 0;
            while (rs.next())
                resCount ++;

            if (logline.resSize != resCount)
                logline.setResSize(resCount);

            rs.close();
            ps.close();
            conn.close();
        } catch (SQLException e) {
            System.out.println("error while executing query " + query);
        }
    }

    public static void executeQuery (LogLine logline) {
        ArrayList<String> res = new ArrayList<>();
        Map<String, String> sqlStatements = new HashMap<>();
        String query = logline.query.statement;
        for(Pair<DBObject, String> comp:logline.query.including_components){
            sqlStatements.putAll(comp.getValue0().getSelectCTIDStatement(query, comp.getValue1()));
        }

        for (Map.Entry<String,String>  comp_and_sql : sqlStatements.entrySet()) {
//            System.out.println(comp_and_sql.getValue());
            try {
                Connection conn = DatabaseRepo.getDefaultPostgresqlConnection();
                PreparedStatement ps = conn.prepareStatement(comp_and_sql.getValue());

                ResultSet rs = ps.executeQuery();
                if (!rs.isBeforeFirst() ) {
                    if(logline.resSize != 0) {
                        System.out.println("No data");
//                        System.out.println(comp_and_sql.getValue());
                        writeLog(String.format("No data for query %s and reference %s",
                                comp_and_sql.getValue(), comp_and_sql.getKey()));
                    }
                }

                ArrayList<String> bnums = new ArrayList<>();
                while (rs.next()) {
                    bnums.add(rs.getString(1));
                }


                for(String num : new HashSet<>(bnums)){
                    res.add(comp_and_sql.getKey() + "_" + num);
                }


                rs.close();
                ps.close();
                conn.close();
            } catch (SQLException e) {
                System.out.println("error while executing query " + logline.seqNUm);
                writeLog(String.format("Error while executing query %s and reference %s.details:\n%s",
                        comp_and_sql.getValue(), comp_and_sql.getKey(), e.getMessage()));
                continue;
            }catch (OutOfMemoryError oe){
                System.out.println("Out of memory error while executing query " + logline.seqNUm);
                writeLog(String.format("Out of memory error while executing query %s and reference %s.details:\n%s",
                        comp_and_sql.getValue(), comp_and_sql.getKey(), oe.getMessage()));
                continue;
            }
        }
        logline.query.setResultSet(res);
    }

    private static void writeLog(String report) {
        try {
            System.out.println(report);
            log.write(report);
            log.newLine();
            log.flush();
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }
}
