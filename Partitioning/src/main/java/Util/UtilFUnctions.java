package Util;

import backend.disk.PartitionManager;
import backend.util.AffinityMatrix;
import backend.util.FlexibleQuadTree;
import backend.util.LogLine;
import me.tongfei.progressbar.ProgressBar;

import org.apache.commons.lang3.StringUtils;
import org.javatuples.Pair;

import javax.mail.Part;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.Timestamp;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static Util.ConfigParams.schema_name;

public class UtilFUnctions {

    public static <T> Pair<T, T> getExtremesWithType(Pair<Object, Object> ex_obj){
        Object value = ex_obj.getValue0();
        if (value instanceof Number) {
            return (Pair<T, T>)
                    new Pair<>(Double.valueOf(String.valueOf(value)), Double.valueOf(String.valueOf(ex_obj.getValue1())));
        }else if(value instanceof Date) {
//            DateFormat df = new SimpleDateFormat("yyyy-mm-dd'T'hh:mm:ss.ssss'Z'");
            return (Pair<T, T>) new Pair<>((Timestamp) value, (Timestamp) ex_obj.getValue1());
        }
        System.out.println("could not convert the extremes");
        return (Pair<T, T>) ex_obj;
    }

    public static boolean exactMatch(String source, String subItem){
        String pattern = "\\b"+subItem+"\\b";
        Pattern p=Pattern.compile(pattern);
        Matcher m=p.matcher(source);
        return m.find();
    }

    public static boolean doesMatch(String source, String subItem){
        Pattern p=Pattern.compile(subItem);
        Matcher m=p.matcher(source);
        return m.find();
    }

    public static boolean exactMatch(ArrayList<String> sourceList, String subItem){
        for(String source : sourceList){
            if (exactMatch(source, subItem))
                return true;
        }
        return false;
    }

    public static String getCompleteTableNameRgx(String table_name){
        return  (schema_name.length() > 0) ? ("\\[" + schema_name + "\\]\\.\\[" + table_name + "\\]") : "\\b" + table_name + "\\b";
    }

    public static String getCompleteTableName(String table_name){
        return  (schema_name.length() > 0) ? ("[" + schema_name + "].[" + table_name + "]") : table_name;
    }

    public static <T> int intersection(ArrayList<T> list1, ArrayList<T> list2) {
        ArrayList<T> list = new ArrayList<T>();

        for (T t : list1) {
            if (list2.contains(t)) {
                list.add(t);
            }
        }

        return list.size();
    }

    public static <T> ArrayList<T> union(ArrayList<T> list1, ArrayList<T> list2) {
        Set<T> set = new HashSet<T>();

        set.addAll(list1);
        set.addAll(list2);

        return new ArrayList<T>(set);
    }

    public static String getComponentName(String query, String comp) {
        String pattern = "\\b" + comp + " as ([0-9a-z_]+)\\b";
        Pattern p = Pattern.compile(pattern);
        Matcher m = p.matcher(query);
        if (m.find()) {
            String name = m.group(1);
            return  (name.equals("where")) ? "" : name;
        }
        pattern = "\\b" + comp + " ([0-9a-z_]+)\\b";
        p = Pattern.compile(pattern);
        m = p.matcher(query);
        if (m.find()) {
            String name = m.group(1);
            return  (name.equals("where") || name.equals("limit")) ? "" : name;
        }
//        System.out.println("return empty for " + comp + " in " + query + ".");
        return "";
    }

    public static ArrayList<Integer> getPartitionNumbers(Set<String> partition_list) {
        ArrayList<Integer> res = new ArrayList<>();
        for(String s:partition_list){
            res.add(Integer.parseInt(StringUtils.substring(s, 1, s.length())));
        }
        return res;
    }

    public static ArrayList<FlexibleQuadTree.Rectangle2D> createNaviTiles(double scale) {
        ArrayList<FlexibleQuadTree.Rectangle2D> res = new ArrayList<>();
        for(double dec = ConfigParams.dec_extreme[0]; dec < ConfigParams.dec_extreme[1]; dec += scale){
            for(double ra = ConfigParams.ra_extreme[0]; ra < ConfigParams.ra_extreme[1]; ra += scale){
                res.add(new FlexibleQuadTree.Rectangle2D(
                        ra,
                        dec,
                        Math.min(ra+scale, ConfigParams.ra_extreme[1]),
                        Math.min(dec+scale, ConfigParams.dec_extreme[1]))
                );
            }
        }
        System.out.println("tile list size for scale = " + scale + " is " + res.size());
        return res;
    }

    public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> unsortMap) {

        List<Map.Entry<K, V>> list =
                new LinkedList<Map.Entry<K, V>>(unsortMap.entrySet());

        Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        Map<K, V> result = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }

        return result;

    }

    public static int getDirectionIndex(String dir) {
        return dirMap.get(dir);
    }

    public static String getIndexDirection(int idx) {
        return mapDir.get(idx);
    }

    public static ArrayList<LogLine> getQueries(String file_path){
        return getQueries(file_path, "\\|\\|");    
    }

    public static ArrayList<LogLine> getQueries(String file_path, String sep) {
        ArrayList<LogLine> logs = new ArrayList<>();
        try
        {
            FileInputStream fis=new FileInputStream(file_path);
            Scanner sc=new Scanner(fis);
            int count = 1;
            String line = sc.nextLine();
            String[] splited_line = line.split(sep);
            int line_size = splited_line.length;

            Path file = Paths.get(file_path);
            long line_count = Files.lines(file).count();

            try (ProgressBar pb = new ProgressBar("Reading queries from CSV file:", line_count)) {

                while(sc.hasNextLine()){
                    line = sc.nextLine();
                    pb.step();
                    splited_line = line.split(sep);
                    LogLine log;
                    if (splited_line[0].contains(":")){
                        log = new LogLine(count, Arrays.copyOfRange(splited_line, 0, line_size));
                    }
                    else {
                        log = new LogLine(count, Arrays.copyOfRange(splited_line, 1, line_size));
                    }
                    logs.add(log);
                    count ++;
                }
            }
            sc.close();
        }
        catch(IOException e)
        {
            System.out.println("catch i/o exception");
            e.printStackTrace();
        }
        return logs;
    }

    public static String[] rsplit1(String text, String delimiter) {
        int lastIndex = text.lastIndexOf(delimiter);
        
        if (lastIndex != -1) {
            String beforeLast = text.substring(0, lastIndex);
            String afterLast = text.substring(lastIndex + delimiter.length());
            return new String[] { beforeLast, afterLast };
        } else {
            return new String[] { text };
        }
    }

    public static int getWordFrequency(String searchText, String targetWord) {
        Pattern pattern = Pattern.compile("\\b%s(?!\\w)".format(targetWord));
        Matcher matcher = pattern.matcher(searchText);
        int wordCount = 0;
        while (matcher.find())
            wordCount++;
        return wordCount;
    }

    public static void StoreDetailedQueryLog(ArrayList<LogLine> queries) {
        DateFormat dateFormat = new SimpleDateFormat("MM/dd/yyyy hh:mm:ss a");
        String outLine = "seqNum||theTime||clientIP||row||statement||resultBlock\n";
        for(LogLine logline : queries){
            outLine = outLine + logline.seqNUm + "||" + dateFormat.format(logline.theTime) + "||" + logline.clientIP + "||" + logline.resSize + "||"
                    + logline.query.statement + "||" + String.join(",", logline.query.result_set
                    + "\n");
        }
        try {
            Path path = Paths.get(ConfigParams.baseDirectoryString + "prefetching/new_workload_WB.txt");
            Files.writeString(path, outLine, StandardCharsets.UTF_8);
        }
        catch (IOException ex) {
            System.out.print("Invalid Path");
        }
    }

    public static void storeQueriesToFile(ArrayList<LogLine> queries, String fileName) {
        try{
            storeObjToFile(queries, fileName);
        }catch(Exception e){System.out.println(e);}
    }

    private static void storeObjToFile(Object obj, String fileName) throws IOException {
        FileOutputStream fout=new FileOutputStream(fileName);
        ObjectOutputStream out=new ObjectOutputStream(fout);
        out.writeObject(obj);
        out.flush();
        out.close();
    }

    public static void storePartitionManagerToFile(PartitionManager pManager, String fileName) {
        try{
            storeObjToFile(pManager, fileName);
        }catch(Exception e){System.out.println(e);}
    }

    public static PartitionManager readPartitionManagerFromFile(String fileName) {
        PartitionManager pManager = new PartitionManager();
        try{
            ObjectInputStream in=new ObjectInputStream(new FileInputStream(fileName));
            pManager = (PartitionManager) in.readObject();
            in.close();
        }catch(Exception e){System.out.println(e);}
        System.out.println("successfully read partition manager");
        return pManager;
    }

    public static void storeAffMatrixToFile(AffinityMatrix aMatrix, String fileName) {
        try{
            storeObjToFile(aMatrix, fileName);
        }catch(Exception e){System.out.println(e);}
    }

    public static AffinityMatrix readAffMatrixFromFile(String fileName) {
        AffinityMatrix aMatrix = new AffinityMatrix();
        try{
            ObjectInputStream in=new ObjectInputStream(new FileInputStream(fileName));
            aMatrix = (AffinityMatrix) in.readObject();
            in.close();
        }catch(Exception e){System.out.println(e);}
        System.out.println("successfully read the Affinity Matrix");
        return aMatrix;
    }

    public static ArrayList<LogLine> readQueriesFromFile(String fileName) {
        ArrayList<LogLine> res = new ArrayList<>();
        try{
            ObjectInputStream in=new ObjectInputStream(new FileInputStream(fileName));
            res = (ArrayList<LogLine>) in.readObject();
            in.close();
        }catch(Exception e){System.out.println(e);}
        return res;
    }

    private static void writeLog(BufferedWriter log, String report) {
        try {
            System.out.println(report);
            log.write(report);
            log.newLine();
            log.flush();
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    public static Map<String, Double> sortByValue(Map<String, Double> unsortMap, final boolean order) {
//        ASC: order = true, DESC: order = false
        List<Map.Entry<String, Double>> list = new LinkedList<>(unsortMap.entrySet());
        list.sort((o1, o2) -> order ? o1.getValue().compareTo(o2.getValue()) == 0
                ? o1.getKey().compareTo(o2.getKey())
                : o1.getValue().compareTo(o2.getValue()) : o2.getValue().compareTo(o1.getValue()) == 0
                ? o2.getKey().compareTo(o1.getKey())
                : o2.getValue().compareTo(o1.getValue()));
        return list.stream().collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (a, b) -> b, LinkedHashMap::new));

    }
}
