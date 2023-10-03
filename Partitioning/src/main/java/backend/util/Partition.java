package backend.util;

import Util.ConfigParams;
import Util.UtilFUnctions;
import backend.disk.TileManager;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Partition implements Serializable{
    public String partition_id;
    public ArrayList<String> tiles;
    public ArrayList<String> query_set;
    public ArrayList<String> grouped_query_set;

    public Partition(String p_id) {
        partition_id = p_id;
        tiles = new ArrayList<>();
        query_set = new ArrayList<>();
        grouped_query_set = new ArrayList<>();
    }

    public Partition(String partition_id, ArrayList<String> tiles, ArrayList<String> query_set, ArrayList<String> grouped_query_set) {
        this.partition_id = partition_id;
        this.tiles = tiles;
        this.query_set = query_set;
        this.grouped_query_set = grouped_query_set;
    }

    public void addToTileList(String tile_id) {
        tiles.add(tile_id);
    }

    public void addToQuerySet(ArrayList<String> queries) {
        query_set = UtilFUnctions.union(query_set, queries);
    }
    public void addToGroupedQuerySet(ArrayList<String> queries) {
        grouped_query_set = UtilFUnctions.union(grouped_query_set, queries);
    }

    public ArrayList<String> getRetrievePartitionStatement(TileManager tileManager){
        ArrayList<String> statements = new ArrayList<>();
        Map<String, ArrayList<FlexibleQuadTree.Rectangle2D>> table_and_bounds = new HashMap<>();
        for (String tile: tiles){
            Tile t = tileManager.getTile(tile);
            ArrayList<FlexibleQuadTree.Rectangle2D> temp = new ArrayList<>();
            if(table_and_bounds.containsKey(t.tile_id[0]))
                temp = table_and_bounds.get(t.tile_id[0]);
            temp.add(t.bound);
            table_and_bounds.put(t.tile_id[0], temp);
        }

        for(String table: table_and_bounds.keySet()){
            TableInfo tableInfo = ConfigParams.tablesInfo.get(table);
            StringBuilder stmt = new StringBuilder(tableInfo.getSelectStatement() + " FROM " + tableInfo.getFromStatement());
            boolean isFirst = true;
            for(FlexibleQuadTree.Rectangle2D bound: table_and_bounds.get(table)){
                if(isFirst){
                    stmt.append(" WHERE (").append(tableInfo.getWhereStatement(bound)).append(" )");
                    isFirst = false;
                    continue;
                }
                stmt.append(" OR (").append(tableInfo.getWhereStatement(bound)).append(" )");
            }
            stmt.append(";");
            statements.add(String.valueOf(stmt));
        }
        return statements;
    }

    public double getPartitionLoad(AffinityMatrix aMatrix, int k){
        double load = 0;
        for(String tile:tiles){
            AffinityMatrix.AffinityEntry tile_affinityEntry = aMatrix.getTileAffinities(tile);
            if(tile_affinityEntry == null)
                continue;
//            load += tile_affinityEntry.accessCount;
            for(Map.Entry<String, Double> freq:tile_affinityEntry.freqs.entrySet()){
                if (tiles.contains(freq.getKey()))
                    continue;
                load += (k * (double)freq.getValue());
            }
        }
        return load;
    }

    public String getHottestBlock(AffinityMatrix aMatrix){
        double maxExitFreq = 0;
        String res = "";
        for(String tile:tiles){
            AffinityMatrix.AffinityEntry tile_affinityEntry = aMatrix.getTileAffinities(tile);
            double exitFreq = 0;
            if(tile_affinityEntry == null)
                continue;
//            load += tile_affinityEntry.accessCount;
            for(Map.Entry<String, Double> freq:tile_affinityEntry.freqs.entrySet()){
                if (tiles.contains(freq.getKey()))
                    continue;
                exitFreq += freq.getValue();
            }
            if(exitFreq > maxExitFreq){
                maxExitFreq = exitFreq;
                res = tile;
            }
        }
        return res;
    }

    @Override
    public String toString() {
        return
//                "Partition{" +
                "{" + partition_id +
//                ", grouped_query_set=" + grouped_query_set +
                "," + tiles +
                '}';
    }

    public int getSize() {
        return tiles.size();
    }


    public void removeTile(String tid) {
        tiles.remove(tid);
    }
}
