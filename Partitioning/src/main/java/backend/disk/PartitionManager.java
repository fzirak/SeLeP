package backend.disk;

import Util.UtilFUnctions;
import backend.util.AffinityMatrix;
import backend.util.Partition;
import backend.util.Tile;
import org.javatuples.Pair;

import javax.mail.Part;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static Util.UtilFUnctions.sortByValue;

public class PartitionManager implements Serializable{
    public Map<String, Partition> partitions;
    public ArrayList<String> non_empty_partitions;
    public Map<String, Double> loads;
    public int increasingIndex;


    public PartitionManager(){
        partitions = new HashMap<>();
        loads = new HashMap<>();
        non_empty_partitions = new ArrayList<>();
        increasingIndex = 0;
    }

    public int incrementIndex(){
//        System.out.println("in increment index");
        increasingIndex++;
        return increasingIndex;
    }

    public void decrementIndex() {
        increasingIndex--;
    }

    public void addPartitionWithLoad(Partition p, double l){
        addPartition(p);
        loads.put(p.partition_id, l);
    }

    public void addPartition(Partition p){
        if(partitions.containsKey(p.partition_id)){
            System.out.println(">> Illegal. Duplicate partition id!!");
        }
        partitions.put(p.partition_id, p);
    }

    public void deletePartition(Partition partition) {
        partitions.remove(partition.partition_id);
    }

    public void updateLoads(ArrayList<String> partitions_id, AffinityMatrix aMatrix, int k){
        System.out.println("updating loads");
        for(String pid:partitions_id){
            if (!partitions.containsKey(pid)){
                System.out.println(pid);
            }
            loads.put(pid, partitions.get(pid).getPartitionLoad(aMatrix, k));
        }
//        System.out.println(loads);
    }

    public void addNewPartition(Partition p){
        partitions.put(p.partition_id, p);
        if (p.grouped_query_set.size() > 0)
            non_empty_partitions.add((p.partition_id));
    }

    public void addTileToPartition(String partition_id, String tile_id) {
        partitions.get(partition_id).addToTileList(tile_id);
    }

    public ArrayList<String> getPartitionQuerySet(String partition_id) {
        return partitions.get(partition_id).query_set;
    }

    public void addToPartitionQuerySet(String partition_id, ArrayList<String> query_list) {
        partitions.get(partition_id).addToQuerySet(query_list);
        if(query_list.size() > 0 && !non_empty_partitions.contains(partition_id))
            non_empty_partitions.add(partition_id);
    }

    public void addToPartitionGroupedQuerySet(String partition_id, ArrayList<String> query_list) {
        partitions.get(partition_id).addToGroupedQuerySet(query_list);
        if(query_list.size() > 0 && !non_empty_partitions.contains(partition_id))
            non_empty_partitions.add(partition_id);
    }

    public ArrayList<String> getPartitionGroupedQuerySet(String partition_id) {
        return partitions.get(partition_id).grouped_query_set;
    }

    public void updatePartitionWithTile(String partition_id, Tile tile) {
        addTileToPartition(partition_id, tile.str_id);
        addToPartitionQuerySet(partition_id, tile.query_set);
        addToPartitionGroupedQuerySet(partition_id, tile.grouped_query_set);
        if(tile.grouped_query_set.size() > 0 && !non_empty_partitions.contains(partition_id))
            non_empty_partitions.add(partition_id);
    }

    public Pair<String, Double> findMostIntersectingPartition(ArrayList<String> grouped_query_list) {
        Pair<String, Double> res = new Pair<String, Double>("-", (double) 0);
        if(grouped_query_list.size() == 0)
            return res;
        for (String partition_id : non_empty_partitions) {
            int intersection = UtilFUnctions.intersection(grouped_query_list, partitions.get(partition_id).grouped_query_set);
            if (grouped_query_list.size() > 0 && (double) ((double)intersection / (double)grouped_query_list.size()) > res.getValue1()) {
                res = new Pair<String, Double>(partition_id, (double) ((double)intersection / (double)grouped_query_list.size()));
            }
        }
        return res;
    }

    public void clear() {
        partitions.clear();
        non_empty_partitions.clear();
    }

    public boolean containsPid(String partition_id) {
        return partitions.containsKey(partition_id);
    }

    public boolean detectOverload(double maxPartitionLoad) {
        for(String pid:partitions.keySet()){
            if (!loads.containsKey(pid)){
                System.out.println("null value for load!!");
                continue;
            }
            if(loads.get(pid) > maxPartitionLoad) {
                System.out.println("detect overload returned true");
                return true;
            }
//            System.out.println(maxPartitionLoad);
//            System.out.println(loads.get(pid));
        }
        System.out.println("detect overload returned false");
        return false;
    }

    public Partition getMostLoadedPartition() {
        double maxLoad = 0;
        String overloadedPartition = "";
        for(String pid:partitions.keySet()){
            if (!loads.containsKey(pid)){
                System.out.println("null value for load!!");
                continue;
            }
            if(loads.get(pid) > maxLoad) {
                maxLoad = loads.get(pid);
                overloadedPartition = pid;
            }
        }
        return partitions.get(overloadedPartition);
    }

    public HashMap<String, Double> getOverloadPartitions(double maxParLoad) {
        HashMap<String, Double> partitionLoad = new HashMap<>();
        String overloadedPartition = "";
        for(String pid:partitions.keySet()){
            if (!loads.containsKey(pid)){
                System.out.println("null value for load!!");
                continue;
            }
            if(loads.get(pid) > maxParLoad) {
                partitionLoad.put(pid, loads.get(pid));
            }
        }
        return (HashMap<String, Double>) sortByValue(partitionLoad);
    }

    public void removeTileFromPartition(String pid, String tid) {
        partitions.get(pid).removeTile(tid);
    }

    public String getLeastFilledPartition() {
        int leastCap = 99999;
        String resPartition = "";
        for(Partition p : partitions.values()){
            if (p.getSize() < leastCap){
                leastCap = p.getSize();
                resPartition = p.partition_id;
            }
        }
        return resPartition;
    }

    public int getNumEmptyPartitions() {
        int count = 0;
        for(Map.Entry<String, Partition> pp : partitions.entrySet()){
            if(pp.getValue().getSize() == 0)
                count ++;
        }
        return count;
    }
}
