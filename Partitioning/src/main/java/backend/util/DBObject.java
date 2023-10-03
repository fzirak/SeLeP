package backend.util;

import Util.ConfigParams;
import Util.UtilFUnctions;
import org.apache.commons.lang3.StringUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class DBObject implements Serializable {
    public String name;
    public DBObjectType type;
    public ArrayList<DBObject> referencedObjs;

    public DBObject(String name, DBObjectType type) {
        this.name = name;
        this.type = type;
        referencedObjs = new ArrayList<>();
    }

    public void setReferencedObjs(ArrayList<DBObject> referencedObjs) {
        this.referencedObjs = referencedObjs;
    }

    public void addRefObject(DBObject obj){
        this.referencedObjs.add(obj);
    }

    public Map<String, String> getSelectCTIDStatement(String query, String instanceName) {
        /*
        Add obj.ctid to select stmt. It is not a general approach
            and cannot work if there is nested queries or group by stmt.

        String query_lower = query.toLowerCase();
        int select_count = UtilFUnctions.getWordFrequency(query_lower, "select");
        int group_by_count = UtilFUnctions.getWordFrequency(query_lower, "group by");
        String sql = "select distinct (far_b.ctid::text::point)[0]::bigint as block_number from " +
                    "( " + query + " ) as far_a " +
                    "natural join " + this.name + " as far_b;";
        */

        instanceName = (instanceName.equals("")) ? name : instanceName;
        Map<String, String> refs_and_stmts = new HashMap<>();
        if (type == DBObjectType.table || type == DBObjectType.view) {
//            String sql = query.replaceFirst("\\b[S|s][E|e][L|l][E|e][C|c][T|t]\\b",
//                    String.format("select (%s.ctid::text::point)[0]::bigint as block_number, ", instanceName));

            String sql;
            if (query.matches("^\\s*\\b[S|s][E|e][L|l][E|e][C|c][T|t]\\b\\s+\\b[D|d][I|i][S|s][T|t][I|i][N|n][C|c][T|t]\\b.*")) {
                sql = query.replaceFirst(
                        "\\b[S|s][E|e][L|l][E|e][C|c][T|t]\\b\\s+\\b[D|d][I|i][S|s][T|t][I|i][N|n][C|c][T|t]\\b",
                        String.format("select distinct (%s.ctid::text::point)[0]::bigint as block_number, ", instanceName));
            } else {
                sql = query.replaceFirst("\\b[S|s][E|e][L|l][E|e][C|c][T|t]\\b",
                        String.format("select (%s.ctid::text::point)[0]::bigint as block_number, ", instanceName));
            }

            // I am not using the distinct keyword here to be able to test the accuracy of the ctid extraction.
            // Later, I will use a hashset to remove duplicates
            sql = "select block_number from  (" + sql + ") as far_c order by block_number;";

            /*
             Again, not general as it is assumed
             each view references just one table
            */
            if(type == DBObjectType.table )
                refs_and_stmts.put(name, sql);
            else{
                if(referencedObjs.size() > 1)
                    System.out.println("! view references more than one table !");
                refs_and_stmts.put(referencedObjs.get(0).name, sql);
            }
            return refs_and_stmts;
        }
        for(DBObject ref:referencedObjs){
            if (ref.name.equals("-")){
                return refs_and_stmts;
            }
            String sql = "select (far_b.ctid::text::point)[0]::bigint as block_number from " +
                    "( " + query + " ) as far_a " +
                    "natural join " + ref.name + " as far_b;";
            refs_and_stmts.put(ref.name, sql);
        }
        return refs_and_stmts;
    }

    @Override
    public String toString() {
        StringBuilder refs = new StringBuilder();
        for(DBObject ref:referencedObjs){
            refs.append(",").append(ref.name);
        }
        return "DBObject{" +
                "name='" + name + '\'' +
                ", type=" + type +
                ", referencedObjs={" + refs +
                "}}";
    }
}
