package backend.util;

import Util.ConfigParams;
import Util.UtilFUnctions;
import org.javatuples.Pair;

import java.io.Serializable;
import java.util.*;

public class QueryInfo implements Serializable {
    public String statement;
    public ArrayList<String> result_set;
    public ArrayList<Pair<DBObject, String>> including_components;
    public Map<String, ArrayList<Pair<String, String>>> including_tables_and_views;

    public QueryInfo(String statement) {
        this.statement = statement;
        this.result_set = new ArrayList<>();
        this.including_components = getIncludingComponents();
//        this.including_tables_and_views = getIncludingTablesAndViews();
    }

    public QueryInfo(String statement, ArrayList<String> resultSet) {
        this.statement = statement;
        this.result_set = resultSet;
        this.including_components = getIncludingComponents();
//        this.including_tables_and_views = getIncludingTablesAndViews();
    }

    public QueryInfo(String statement, String resSetArray) {
        this.statement = statement;
        this.result_set = new ArrayList<>();
        resSetArray = resSetArray.replaceAll("'", "");
        String[] resArreySplitted = (resSetArray.replaceAll("\\[|\\]|\\s", "").split(","));
        Collections.addAll(result_set, resArreySplitted);
        // TODO: Next line is commented for tpcds dataset!!!
        // this.including_components = getIncludingComponents();
//        this.including_tables_and_views = getIncludingTablesAndViews();
    }

    public ArrayList<Pair<DBObject, String>> getIncludingComponents() {
        String query_lc = statement.toLowerCase();
        ArrayList<Pair<DBObject, String>> comps = new ArrayList<>();
        ArrayList<String> comps_name = new ArrayList<>();
        for (DBObject comp: ConfigParams.db_comps.components.values()) {
            if (UtilFUnctions.doesMatch(query_lc, " " + comp.name + "((\\s)+|(\\,)+|(\\(([0-9a-z\\.\\,\\s]+)\\)))")
//                    || UtilFUnctions.doesMatch(query_lc, " " + comp.name + " ")
            ){
                /*
                    No need to consider function references
                */
//                if(comp.type != DBObjectType.table && comp.type != DBObjectType.view) {
//                    for (DBObject ref : comp.referencedObjs) {
//                        String name = UtilFUnctions.getComponentName(query_lc, ref.name);
//                        comps.add(new Pair<>(ref, name));
//                        comps_name.add(ref.name);
//                    }
//                    comps_name.add(comp.name);
//                    continue;
//                }
                String name = UtilFUnctions.getComponentName(query_lc, comp.name);
                if(name.equals("from") || name.equals("select")){
                    System.out.println(String.format("something wierd happened. In %s, the instance name for %s is from", statement, comp));
                    continue;
                }
                comps.add(new Pair<>(comp, name));
                comps_name.add(comp.name);
            }
        }
        ArrayList<String> view_refs = new ArrayList<>();
        for(Pair<DBObject, String> compn:comps ){
            DBObject comp = compn.getValue0();
            if (comp.type != DBObjectType.view) {
                for(DBObject refobj: comp.referencedObjs){
                    view_refs.add(refobj.name);
                }
            }
        }
                


        ArrayList<Pair<DBObject, String>> res = new ArrayList<>();
        ArrayList<String> added_comps_name = new ArrayList<>();
        for(Pair<DBObject, String> compn:comps ){
            DBObject comp = compn.getValue0();
            boolean refs_included = true;
            if (comp.type == DBObjectType.view) {
                for (DBObject ref_comp : comp.referencedObjs) {
                    if (!comps_name.contains(ref_comp.name)) {
                        refs_included = false;
                        break;
                    }
                }
                if (refs_included)
                    continue;
            }
            else if (comp.type == DBObjectType.function) {
                for (DBObject ref_comp : comp.referencedObjs) {
                    if (!comps_name.contains(ref_comp.name) && !view_refs.contains(ref_comp.name)) {
                        refs_included = false;
                        break;
                    }
                }
                if (refs_included)
                    continue;
            }
            if(!added_comps_name.contains(comp.name)) {
                res.add(compn);
                added_comps_name.add(comp.name);
            }
        }
        return res;
    }

    public Map<String, ArrayList<Pair<String, String>>> getIncludingTablesAndViews() {
        String query_lc = statement.toLowerCase();
        Map<String, ArrayList<Pair<String, String>>> tableAndViews = new HashMap<>();
        for (TableInfo table_info: ConfigParams.tablesInfo.values()){
            boolean is_included = false;
            ArrayList<Pair<String, String>> views = new ArrayList<>();
            for (String view: table_info.views_name){
                view = view.toLowerCase();
                // TODO: change dbo to schema name
                if (UtilFUnctions.exactMatch(query_lc, " " + view + "(\\s|\\,|(\\([0-9a-z\\.\\,\\s]+\\)))") ||
                        UtilFUnctions.exactMatch(query_lc, "dbo." + view + "(\\s|\\,|(\\([0-9a-z\\.\\,\\s]+\\)))")){
                    String name = UtilFUnctions.getComponentName(query_lc, view);
                    is_included = true;
                    views.add(new Pair<>(view, name));
                }
            }
            if(is_included){
                tableAndViews.put(table_info.table_name, views);
            }
        }
        return tableAndViews;
    }

    public void setResultSet(ArrayList<String> res_set){
        this.result_set = res_set;
    }

    public void addToResultSet(String res){
        if(!result_set.contains(res))
            this.result_set.add(res);
    }

    public boolean containsComp(String Comp_name){
        for(int i=0; i < this.including_components.size(); i++){
            if (including_components.get(i).getValue0().name.equals(Comp_name))
                return true;
        }
        return false;
    }

    public boolean containsKeyword(String keyword) {
        String query_lc = statement.toLowerCase();
        if (query_lc.contains(keyword.toLowerCase())) {
            return true;
        }
        return false;
    }

    @Override
    public String toString() {
        return "QueryInfo{" +
                "statement='" + statement + '\'' +
                ", result_set=" + result_set +
                ", including_comps=" + including_components +
//                ", including_table_and_views=" + including_tables_and_views +
                '}';
    }
}
