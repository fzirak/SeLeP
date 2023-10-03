package backend.Database;

import Util.ConfigParams;
import backend.util.QueryInfo;
import backend.util.TableInfo;
import org.apache.commons.lang3.StringUtils;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static Util.ConfigParams.schema_name;

public class DatabaseUtil {
    public static Map<String, ArrayList<String>> getIncludedTables(String query_lc, String select_clause) {
        Map<String, ArrayList<String>> tableAndAxes = new HashMap<>();
        for (TableInfo table_info: ConfigParams.tablesInfo.values()){
            boolean is_included = false;
            for (String view: table_info.views_name){
                if (query_lc.contains(view.toLowerCase())){
                    is_included = true;
                    break;
                }
            }
            if(is_included){
                System.out.println("found " + table_info.table_name);
                ArrayList<String> cond = new ArrayList<>();
                if (select_clause.toLowerCase().contains("*")) {
                    System.out.println("found *");
                    tableAndAxes.put(table_info.table_name, table_info.axes_names);
                    continue;
                }
                for(String ax: table_info.axes_names) {
                    if (select_clause.toLowerCase().contains(ax.toLowerCase())) {
                        System.out.println("added");
                        cond.add(ax);
                    }
                }
                tableAndAxes.putIfAbsent(table_info.table_name, cond);
            }
        }
        return tableAndAxes;
    }

    public static int executePSQLQuery(String query){
        System.out.println("executing query");
        int resCount = 0;
        try {
            Connection conn = DatabaseRepo.getDefaultPostgresqlConnection();
            PreparedStatement ps = conn.prepareStatement(query);

            ResultSet rs = ps.executeQuery();
            while (rs.next()) {
                resCount++;
            }

            rs.close();
            ps.close();
            conn.close();
        } catch (SQLException e) {
            System.out.println("error while executing query " + query);
            System.out.println(e.getMessage());
        }
        return resCount;
    }
}
