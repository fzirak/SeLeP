package Configuration;

import Util.ConfigParams;
import backend.Database.DatabaseRepo;
import backend.util.DBComponents;
import backend.util.DBObject;
import backend.util.DBObjectType;
import backend.util.TableInfo;

import java.io.*;
import java.util.*;

public class Config {
    public Config() {}

    public Config(boolean isNavigation, boolean isOnServer) {
        this.setConfig(isNavigation, isOnServer);
    }


    public void setConfig(boolean isNavigation, boolean isOnServer){
        ConfigParams.host = "127.0.0.1";

        /* SQL Server databases ----------------
            ConfigParams.user = "sa";
            ConfigParams.password = "pass";
            ConfigParams.port = "1433";
            ConfigParams.dbname = "MyBestDR7";
            ConfigParams.schema_name = "";

            ConfigParams.dbname = "sqlshare";
            ConfigParams.schema_name = "1059";
        ------------------------------------------ */

        /* PostgreSQL databases ----------------
            DBInterface.dbname = "birds_sqlshare";

            ConfigParams.dbname = "sdss_1";
            ConfigParams.user = "farzaneh";
            ConfigParams.password = "pj1_1t";
            ConfigParams.port = "5432";
            ConfigParams.schema_name = "";
        ------------------------------------------ */

        if(isOnServer){
            ConfigParams.dbname = "sdss_1"; //tpcds
            ConfigParams.user = "postgres";
            ConfigParams.password = "pass";
            ConfigParams.baseDirectoryString = "Documents/";
        }
        else{
            ConfigParams.dbname = "sdss_2";
            ConfigParams.user = "farzaneh";
            ConfigParams.password = "pass";
            ConfigParams.baseDirectoryString = "/Implementation/";
        }
        
        ConfigParams.port = "5432";
        ConfigParams.schema_name = "";

        if (isNavigation){
            ConfigParams.n_candidate_partitions = 5;
            ConfigParams.cache_size = 20;
            ConfigParams.prefetch_cache_size = 5;
            ConfigParams.n_query_sequence = 1;
            ConfigParams.learning_rate = 0.4;
            ConfigParams.intersection_precision = 0.3;
            ConfigParams.ra_extreme = new double[]{0, 360};
            ConfigParams.dec_extreme = new double[]{-24, 84};
            ConfigParams.scales = new double[]{30, 10, 2.5, 0.25};
            ConfigParams.navi_query_file_path = ConfigParams.baseDirectoryString + "prefetching/naviTrainLog.txt";
            ConfigParams.navi_test_queries_path = ConfigParams.baseDirectoryString + "prefetching/naviTestLog.txt";
        }
        else {
            ConfigParams.n_candidate_partitions = 10;
            ConfigParams.cache_size = 18;
            ConfigParams.prefetch_cache_size = 10;
            ConfigParams.n_query_sequence = 1;
            ConfigParams.learning_rate = 0.2;
            ConfigParams.intersection_precision = 0.125;
            ConfigParams.query_file_path = ConfigParams.baseDirectoryString + "prefetching/queriesWithBlocks.txt";
            ConfigParams.test_queries_path = ConfigParams.baseDirectoryString + "prefetching/test_queries1.txt";
        }

        /* list of table name(s), coordinate column(s) name, max num of element per tile in that tree */
        ConfigParams.db_comp_file = ConfigParams.baseDirectoryString + "prefetching/DBComps.txt";
        ConfigParams.db_comps = new DBComponents();
        setUpDBComps();
    }

    public void setUpDBComps() {
        File file = new File(ConfigParams.db_comp_file);
        HashMap<String, DBObject> comps = new HashMap<>();
        try {
            FileInputStream fis = new FileInputStream(file);
            Scanner sc=new Scanner(fis);
            String line;
            String[] line_data;
            int lineCount = 0;
            DBObjectType type = DBObjectType.unknown;
//            ArrayList<DBObject> comps = new ArrayList<>();
            while(sc.hasNextLine()){
                line = sc.nextLine().replaceAll("\\s", "").toLowerCase();
                if (line.contains("$")){
                    type = DBObjectType.getType(line.split("\\$")[1]);
                    continue;
                }
                line_data = line.split(",");
                DBObject new_obj = new DBObject(line_data[0], type);
                if (type == DBObjectType.function || type == DBObjectType.view){
                    String[] refs = line_data[1].split("\\|");
                    for (String ref: refs) {
                        DBObject ref_obj = comps.get(ref);
                        if (ref_obj == null){
                            ref_obj = new DBObject(ref, DBObjectType.unknown);
                            System.out.println(String.format(">> Undefined reference %s for %s", new_obj.name, ref_obj.name));
                        }
                        new_obj.addRefObject(ref_obj);
                    }
                }
                comps.put(new_obj.name, new_obj);
            }
            sc.close();
            fis.close();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        ConfigParams.db_comps.setComponents(comps);
        System.out.println(ConfigParams.db_comps);
    }

}
