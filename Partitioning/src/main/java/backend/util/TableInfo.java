package backend.util;

import backend.Database.DatabaseRepo;
import org.apache.commons.lang3.StringUtils;
import org.javatuples.Pair;

import java.util.ArrayList;

import static Util.ConfigParams.schema_name;

public class TableInfo {
    public String table_name;
    public String table_name_with_schema;
    public ArrayList<String> axes_names;
    public ArrayList<String> axes_types;

    public ArrayList<String> views_name;
    public int elem_per_tile;

    public String derived_tb_name = "";
    public  String join_column1;
    public  String join_column2;

    public ArrayList<String> columns;

    public TableInfo(String table_name) {
        this.table_name = table_name;
        this.table_name_with_schema =
                (schema_name.length() > 0) ? ('[' + schema_name + "].[" + table_name + ']') : table_name;
        this.views_name = new ArrayList<>();
        this.columns = DatabaseRepo.getTableColumnNames(table_name_with_schema);
    }

    public void setAxes_names(ArrayList<String> axes_names) {
        this.axes_names = axes_names;
        if(!(derived_tb_name.length() > 0)){
            for(String s: axes_names) {
                if(!columns.contains(s))
                    columns.add(s);
            }
        }
    }

    public void addAxes_name(String axes_name) {
        this.axes_names.add(axes_name);
        if(!(derived_tb_name.length() > 0)){
            if(!columns.contains(axes_name))
                columns.add(axes_name);
        }
    }

    public void setViews_names(ArrayList<String> axes_names) {
        this.views_name = axes_names;
    }

    public void addViews_name(String axes_name) {
        this.views_name.add(axes_name);
    }

    public void setAxes_types(ArrayList<String> axes_types) {
        this.axes_types = axes_types;
    }

    public void addAxes_type(String axes_type) {
        this.axes_types.add(axes_type);
    }

    public void setElem_per_tile(int elem_per_tile) {
        this.elem_per_tile = elem_per_tile;
    }

    public void setDerived_tb_name(String derived_tb_name) {
        this.derived_tb_name = (schema_name.length() > 0) ?
                ('[' + schema_name + "].[" + derived_tb_name + ']') : derived_tb_name;
    }

    public void setJoin_column1(String join_column1) {
        this.join_column1 = join_column1;
    }

    public void setJoin_column2(String join_column2) {
        this.join_column2 = join_column2;
    }

    public String getTable_name() {
        return table_name;
    }

    public String getSelectStatement(){
        if(derived_tb_name.length() > 0)
            return "SELECT far_a." + String.join(", far_a.", columns) + ", far_b." +
                String.join(", far_b.", axes_names);

        return "SELECT " + String.join(", ", columns);
    }

    public String getFromStatement(){
        String stmt = table_name_with_schema;
        if (derived_tb_name.length() > 0)
            return stmt + " far_a INNER JOIN " + derived_tb_name +
                " far_b ON far_a." + join_column1 + " = far_b." + join_column2;

        return stmt;
    }

    public String getWhereStatement(FlexibleQuadTree.Rectangle2D bound){
        String x = (derived_tb_name.length() > 0) ? "far_b." + axes_names.get(0) : axes_names.get(0);
        String wStmt = x + " > " + bound.getMinX() + " AND " + x + " <= " + bound.getMaxX();
        if (axes_names.size() > 1) {
            String y = (derived_tb_name.length() > 0) ? "far_b." + axes_names.get(1) : axes_names.get(1);
            wStmt = wStmt + " AND " + y + " > " + bound.getMinY() + " AND " + y + " <= " + bound.getMaxY();
        }
        return wStmt;
    }

    public String getTableElementsStatement(){
        return this.getSelectStatement() + " FROM " + this.getFromStatement();
//        return "select " + String.join(", ", columns) + " from " + table_name_with_schema;
    }


    @Override
    public String toString() {
        return "TableInfo{" +
                "table_name='" + table_name + '\'' +
                ", table_name_with_schema='" + table_name_with_schema + '\'' +
                ", axes_names=" + axes_names +
                ", axes_types=" + axes_types +
                ", elem_per_tile=" + elem_per_tile +
                ", views_name=" + views_name +
                '}';
    }

    public boolean checkSelecetClause(String select_clause) {
        for(String ax: axes_names){
            if (!select_clause.contains(ax))
                return false;
        }
        return true;
    }

    public Pair<Integer, Integer> getAxesIndex() {
        int x_index = 0;
        int y_index = 0;
        ArrayList<String> cns = (ArrayList<String>) columns.clone();
        for(String s: axes_names) {
            if(!columns.contains(s))
                cns.add(s);
        }
        x_index = cns.indexOf(axes_names.get(0).toLowerCase());
        y_index = (axes_names.size() > 1) ? cns.indexOf((axes_names.get(1).toLowerCase())) : x_index;
        return new Pair<>(x_index, y_index);
    }

    public String getSelectTileStatement(QueryInfo query) {
        String final_q = getStatementWithAxes(query);
        String sql = "select distinct tile_id from " +
                "( " + final_q + " ) as far_a " +
                "inner join " +  table_name + "_tiles on ($x > min_x and $x <= max_x";
        String xx = (axes_names.get(0).equals("(ramin+ramax)/2")) ? "(far_a.ramin + far_a.ramax)/2" : "far_a." + axes_names.get(0);
        sql = StringUtils.replaceEach(sql, new String[]{"$x"}, new String[]{xx});
        if(axes_names.size() > 1){
            sql = sql + " and $y > min_y and $y <= max_y";
            String yy = (axes_names.get(1).equals("(decmin+decmax)/2")) ? "(far_a.decmin + far_a.decmax)/2" : "far_a." + axes_names.get(1);
            sql = StringUtils.replaceEach(sql, new String[]{"$y"}, new String[]{yy});
        }
        sql = sql + ");";
        return sql;
    }

    public String getStatementWithAxes(QueryInfo query) {
        String final_q = query.statement;
        String q = query.statement.toLowerCase();
        String select_clause = StringUtils.substringBetween(q, "select", "from");
        if(derived_tb_name.length() > 0) {
            final_q = "select far_d." + String.join(", far_d.", axes_names) + " from " + derived_tb_name +
                    " far_d inner join (" + final_q + ") as far_c on (far_c." + join_column1 + " = far_d." +
                    join_column2 + ")";
        } else {
            if (select_clause.contains("*"))
                return final_q;
            for (String ax : axes_names) {
                if (!select_clause.contains(ax)) {
                    String comp_name = query.including_tables_and_views.get(table_name).get(0).getValue1();
                    comp_name = (comp_name.length() > 0) ? comp_name + "." : "";
                    final_q = final_q.replaceFirst("(?i)from", ", " + comp_name + ax + " from");
                }
            }
        }
        return final_q;
    }
}
