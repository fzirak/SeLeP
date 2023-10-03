package Util;

import backend.util.DBComponents;
import backend.util.TableInfo;

import java.util.ArrayList;
import java.util.Map;

public class ConfigParams {
    public static String baseDirectoryString;
    public static String table_info_file;
    public static String view_info_file;
    public static String dbname;//  = "test";
    public static String schema_name;
    public static String user;//  = "testuser";
    public static String password;// = "password";
    public static String host;// = "127.0.0.1";
    public static String port;// = "5432";
    public static Map<String, TableInfo> tablesInfo;
    public static double intersection_precision;
    public static ArrayList<String> tables_name;
    public static String query_file_path;
    public static String test_queries_path;

    public static Map<String, ArrayList<String>> tb_name_ax;// = {"table":(X,Y)};
    public static Map<String, Integer> tile_size;
    public static int cache_size;
    public static int prefetch_cache_size;
    public static int n_candidate_partitions = cache_size - 2;
    public static double learning_rate;
    public static int n_query_sequence = 1;

    public static double[] ra_extreme = {0, 360};
    public static double[] dec_extreme = {-24.2, 85};
    public static double[] scales = {30, 10, 0.5};

    public static String navi_query_file_path;
    public static  String navi_test_queries_path;

    public static String db_comp_file;
    public static DBComponents db_comps;

    public static String serverIP = "127.0.0.1";
    public static int serverPort = 8080;

    public static int logicalBlockSize = 8;
    public static int resSizeLimit = 500; //4096; // this is for ignoring affinities in large result sets

    public static boolean is_navigation = false;
}
