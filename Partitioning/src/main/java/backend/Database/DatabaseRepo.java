package backend.Database;

import Util.ConfigParams;
import backend.util.FlexibleQuadTree;
import backend.util.TableInfo;
import org.apache.commons.lang3.StringUtils;
import org.javatuples.Pair;

import java.math.BigDecimal;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import static Util.ConfigParams.*;

public class DatabaseRepo {
    public static Connection getPostgresqlConnection(String host, String port, String dbname, String user, String password) {
        Connection conn = null;
        try {
            conn = DriverManager.getConnection(
                    "jdbc:postgresql://" + host + ":" + port + "/" + dbname,
                    user,
                    password
            );
        } catch (SQLException e) {
            System.out.println("error opening connection to database '" + dbname + "' as user '" + user + "'");
            e.printStackTrace();
        }
        return conn;
    }

    public static Connection getDefaultPostgresqlConnection() {
        return getPostgresqlConnection(host, port,dbname,user,password);
    }

    public static Connection getMssqlConnection(String host, String port, String dbname, String user, String password) {
        Connection conn = null;
        try {
            conn = DriverManager.getConnection(
                    "jdbc:sqlserver://" + host + ":" + port + ";databaseName=" + dbname +
                            ";encrypt=true;trustServerCertificate=true;",
                    user,
                    password
            );
        } catch (SQLException e) {
            System.out.println("error opening connection to database '" + dbname + "' as user '" + user + "'");
            e.printStackTrace();
        }
        return conn;
    }

    public static Connection getDefaultMssqlConnection() {
        return getMssqlConnection(host,port,dbname,user,password);
    }

    public static void clearMssqlCache(){
        Connection conn;
        conn = getDefaultMssqlConnection();
        try {
            CallableStatement callableStatement = conn.prepareCall("CHECKPOINT;DBCC DROPCLEANBUFFERS;");
            callableStatement.executeUpdate();
            callableStatement.close();
//            PreparedStatement ps = conn.prepareStatement("CHECKPOINT; \n" +
//                    "GO \n" +
//                    "DBCC DROPCLEANBUFFERS; \n" +
//                    "GO");
//            ResultSet rs = ps.executeQuery();
//            int count = 0;
//            while (rs.next()) {
//                count ++;
//            }
//            if(count == 0)
//                System.out.println("count was 0 for " + sql);
//            rs.close();
//            ps.close();
            conn.close();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    public static void cleanDatabasePlanCache(String db_name){
        String stmt = "DECLARE @intDBID INT;\n" +
                "SET @intDBID = (SELECT [dbid] FROM master.dbo.sysdatabases WHERE name = N'" + db_name + "');\n" +
                "DBCC FLUSHPROCINDB (@intDBID);";

        Connection conn;
        conn = getDefaultMssqlConnection();
        try {
            CallableStatement callableStatement = conn.prepareCall(stmt);
            callableStatement.executeUpdate();
            callableStatement.close();
            conn.close();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    public static ArrayList<Long> executeQueries(ArrayList<String> stmts){
        Connection conn;
        conn = getDefaultMssqlConnection();
        ArrayList<Long> times = new ArrayList<>();
        try {
            for(String sql: stmts) {
                PreparedStatement ps = conn.prepareStatement(sql);
                long s = System.currentTimeMillis();
                ResultSet rs = ps.executeQuery();
                long e = System.currentTimeMillis();
                times.add(e-s);
                int count = 0;
                while (rs.next()) {
                    count++;
                }
                if (count == 0)
                    System.out.println("count was 0 for " + sql);
                rs.close();
                ps.close();
            }
                conn.close();
        } catch (SQLException e) {
//                System.out.println(sql);
            throw new RuntimeException(e);
        }
//        System.out.println(times);
        return times;
    }

    public static ArrayList<Long> executeQueries(ArrayList<String> stmts, Map<Integer, ArrayList<String>> prefetched_partitions){
        Connection conn;
        conn = getDefaultMssqlConnection();
        ArrayList<Long> times = new ArrayList<>();
        int n = 0;
        try {
            for(String sql: stmts) {
                PreparedStatement ps = conn.prepareStatement(sql);
                long s = System.currentTimeMillis();
                ResultSet rs = ps.executeQuery();
                long e = System.currentTimeMillis();
                times.add(e-s);
                int count = 0;
                while (rs.next()) {
                    count++;
                }
                if (count == 0)
                    System.out.println("count was 0 for " + sql);
                rs.close();
                ps.close();
                Scanner myObj = new Scanner(System.in);

                executeQueries(prefetched_partitions.get(n));
                n++;
            }
            conn.close();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
        return times;
    }

    public static ArrayList<String> getTableColumnNames(String table_name){
        Connection conn;
        ArrayList<String> headers = new ArrayList<>();
        try {
            conn = getDefaultMssqlConnection();
            PreparedStatement ps = conn.prepareStatement("select top 1 * from " + table_name);
            ResultSet rs = ps.executeQuery();
            //Retrieving the list of column names
            ResultSetMetaData rsMetaData = rs.getMetaData();
            int headerCount = rsMetaData.getColumnCount();
            for (int i = 1; i <= headerCount; i++)
                headers.add(rsMetaData.getColumnName(i).toLowerCase());
            rs.close();
            ps.close();
            conn.close();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
        return headers;
    }

    public static Pair<Integer, Integer> getTableBlockRange(String table_name){
        Connection conn;
        int minb = 0;
        int maxb = 0;
        try {
            conn = getDefaultPostgresqlConnection();
            String query = "select min(a.block_number) as minb, max(a.block_number) as maxb from " +
                    "(select (ctid::text::point)[0]::bigint as block_number from " + table_name + ") a";
            PreparedStatement ps = conn.prepareStatement(query);
            ResultSet rs = ps.executeQuery();
            if (rs.next()) {
                minb = rs.getInt("minb");
                System.out.println(minb);
                maxb = rs.getInt("maxb");
                System.out.println(maxb);
            }
            rs.close();
            ps.close();
            conn.close();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
        return new Pair<>(minb, maxb);
    }

    public static ArrayList<Integer> getTableBlocknumbers(String table_name){
        Connection conn;
        ArrayList<Integer> bNums = new ArrayList<>();
        try {
            conn = getDefaultPostgresqlConnection();
            String query = "select distinct (ctid::text::point)[0]::bigint/" + logicalBlockSize + " as block_number from " + table_name + " order by block_number";
            PreparedStatement ps = conn.prepareStatement(query);
            ResultSet rs = ps.executeQuery();
            while (rs.next()) {
                bNums.add(rs.getInt("block_number"));
            }
            rs.close();
            ps.close();
            conn.close();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
        return bNums;
    }

    public static ArrayList<ArrayList<Object>> getTableDataPoints(TableInfo table){
        Connection conn;
        List<String> headers = new ArrayList<>();
        ArrayList<Object> arr = new ArrayList<>();
        ArrayList<ArrayList<Object>> qResult = new ArrayList<>();
        try {
            conn = getDefaultMssqlConnection();
            PreparedStatement ps = conn.prepareStatement(table.getTableElementsStatement());
            System.out.println("getTableDataPoints :" + table.getTableElementsStatement());
            ResultSet rs = ps.executeQuery();
            //Retrieving the list of column names
            ResultSetMetaData rsMetaData = rs.getMetaData();
            int headerCount = rsMetaData.getColumnCount();
            for (int i = 1; i <= headerCount; i++) {
                headers.add(rsMetaData.getColumnName(i));
            }

            while (rs.next()) {
                arr.clear();
                for (int i = 0; i < headerCount; i++) {
                    arr.add(rs.getObject(headers.get(i)));
                }
                qResult.add((ArrayList<Object>) arr.clone());
            }

            rs.close();
            ps.close();
            conn.close();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
        return qResult;
    }

    public static Pair<Pair<Object, Object>, Pair<Object, Object>> getTableExtremes(TableInfo table_info) {
        Connection conn;
        Pair<Object, Object> x_res;
        Pair<Object, Object> y_res;
        try {
            conn = getDefaultMssqlConnection();
            String sql = "select min($x), max($x)";
            sql = StringUtils.replaceEach(sql, new String[]{"$x"}, new String[]{table_info.axes_names.get(0)});
            if (table_info.axes_names.size() > 1){
                sql = sql + ", min($y), max($y)";
                sql = StringUtils.replaceEach(sql, new String[]{"$y"}, new String[]{table_info.axes_names.get(1)});
            }
            sql = sql + " from " + table_info.getFromStatement();
            System.out.println("getTableExtremes" + sql);
            PreparedStatement ps = conn.prepareStatement(sql);
            ResultSet rs = ps.executeQuery();
            rs.next();
//            System.out.println("getTableExtremes 108");
            x_res = new Pair<>(rs.getObject(1), rs.getObject(2));
//            System.out.println("getTableExtremes 110");
            y_res = (table_info.axes_names.size() > 1) ? new Pair<>(rs.getObject(3), rs.getObject(4)) : x_res;
            rs.close();
            ps.close();
            conn.close();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
        return new Pair<>(x_res, y_res);
    }

    public static Pair<Integer, Integer> getAxesIndex(String table_name, ArrayList<String> axes) {
        Connection conn;
        List<String> headers = new ArrayList<>();
        int x_index = 0;
        int y_index = 0;
        try {
            conn = getDefaultMssqlConnection();
            PreparedStatement ps = conn.prepareStatement("select top 1 * from " + table_name);
            ResultSet rs = ps.executeQuery();
            ResultSetMetaData rsMetaData = rs.getMetaData();
            int headerCount = rsMetaData.getColumnCount();
            for (int i = 1; i <= headerCount; i++) {
                headers.add(rsMetaData.getColumnName(i));
            }
            headers.replaceAll(String::toLowerCase);
            System.out.println("getAxesIndex " + headers);
            x_index = headers.indexOf(axes.get(0).toLowerCase());
            y_index = (axes.size() > 1) ? headers.indexOf((axes.get(1).toLowerCase())) : x_index;
            rs.close();
            ps.close();
            conn.close();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
        return new Pair<>(x_index, y_index);
    }

    public static void createTileTable(String table_name, ArrayList<String> axes, ArrayList<String> types){
        Connection conn;
        try {
            conn = getDefaultMssqlConnection();
            Statement stmt = conn.createStatement();
            table_name = (schema_name.length() > 0) ? ('[' + schema_name + "].[" + table_name + "_tiles]") : table_name + "_tiles";
            String sql = "DROP TABLE IF EXISTS " + table_name + "; " +
                    "CREATE TABLE " + table_name +
                    "(tile_id varchar(250) Primary Key, " +
                    " min_x " + types.get(0) + ", " +
                    " max_x " + types.get(0);
            if(axes.size() > 1){
                sql = sql + ", min_y " + types.get(1) + ", max_y " + types.get(0);
            }
            sql = sql + ");";
            System.out.println("createTileTable " + sql);
            stmt.executeUpdate(sql);
            stmt.close();
            conn.close();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }
    public static void insertTilesIntoTileTableForOne(ArrayList<FlexibleQuadTree.Rectangle2D> tiles, String table_name, ArrayList<String> axes){
        String tb = table_name;
        table_name = (schema_name.length() > 0) ? ('[' + schema_name + "].[" + table_name + "_tiles]") : table_name + "_tiles";
        String sql = "INSERT INTO " + table_name + " (tile_id, min_x, max_x) "
                    + "VALUES(?, ?, ?)";
        try (
                Connection conn = getDefaultMssqlConnection();
                PreparedStatement statement = conn.prepareStatement(sql);) {
            int count = 0;

            for (FlexibleQuadTree.Rectangle2D tile : tiles) {
                statement.setString(1, tb + "_" + count);
                statement.setBigDecimal(2, BigDecimal.valueOf(tile.getMinX()));
                statement.setBigDecimal(3, BigDecimal.valueOf(tile.getMaxX()));

                statement.addBatch();
                count++;
                // execute every 100 rows or less
                if (count % 100 == 0 || count == tiles.size()) {
                    statement.executeBatch();
                }
            }
        } catch (SQLException ex) {
            System.out.println("error" + ex.getMessage());
        }
    }

    public static void insertTilesIntoTileTableForTwo(ArrayList<FlexibleQuadTree.Rectangle2D> tiles, String table_name, ArrayList<String> axes){
        table_name = (schema_name.length() > 0) ? ('[' + schema_name + "].[" + table_name + "_tiles]") : table_name + "_tiles";
        String sql = "INSERT INTO " + table_name + " (tile_id, min_x, max_x, min_y, max_y) "
                + "VALUES(?, ?, ?, ?, ?)";
        try (
                Connection conn = getDefaultMssqlConnection();
                PreparedStatement statement = conn.prepareStatement(sql);) {
            int count = 0;

            for (FlexibleQuadTree.Rectangle2D tile : tiles) {
                statement.setString(1, tile.toString());
                statement.setBigDecimal(2, BigDecimal.valueOf(tile.getMinX()));
                statement.setBigDecimal(3, BigDecimal.valueOf(tile.getMaxX()));
                statement.setBigDecimal(4, BigDecimal.valueOf(tile.getMinY()));
                statement.setBigDecimal(5, BigDecimal.valueOf(tile.getMaxY()));

                statement.addBatch();
                count++;
                // execute every 100 rows or less
                if (count % 100 == 0 || count == tiles.size()) {
                    statement.executeBatch();
                }
            }
        } catch (SQLException ex) {
            System.out.println("error" + ex.getMessage());
        }
    }

    public static String getTypeAndPrecision(String type) {
        switch(type) {
            case "decimal":
                return "decimal(24,8)";
            case "datetime":
                return "datetime";
            default:
                return "decimal(24,8)";
        }

    }
}
