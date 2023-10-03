package backend.util;

import Util.ConfigParams;
import Util.UtilFUnctions;
import backend.disk.NewTileManager;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class AffinityMatrix implements Serializable {
    public Map<String, AffinityEntry> affinities;
    double learning_rate = 0.2;

    public AffinityMatrix() {
        this.affinities = new HashMap<>();
    }

    public AffinityEntry getTileAffinities(String bid){
        return affinities.get(bid);
    }

    @Override
    public String toString() {
        return "AffinityMatrix{" +
                "affinities=" + affinities +
                ", learning_rate=" + learning_rate +
                '}';
    }

    public void updateTileAccess(String blockID, ArrayList<String> result_set, int maxWindowSize) {
        AffinityEntry tile = affinities.get(blockID);
        if(tile == null) {
//            System.out.println("tile is null");
            tile = new AffinityEntry(blockID);
        }
        int totalTiles = result_set.size();
        tile.incrementAccessCount();
        for(String blockID2 : result_set) {
            if(blockID.equals(blockID2))
                continue;
            tile.updateFrequency(blockID2, totalTiles, maxWindowSize);
        }
        affinities.put(blockID, tile);
    }

    public void partialUpdateTileAccess(String blockID, ArrayList<String> result_set, NewTileManager tileManager, int maxWindowSize) {
        AffinityEntry tile = affinities.get(blockID);
        if(tile == null) {
            tile = new AffinityEntry(blockID);
        }
        tile.incrementAccessCount();
        NewTile tileObj = tileManager.getTile(blockID);
        String id = tileObj.partition_id;
        for(String blockID2 : result_set) {
            NewTile tileObj2 = tileManager.getTile(blockID2);
            if(blockID.equals(blockID2) ||  tileObj2 == null || !Objects.equals(tileObj2.partition_id, tileObj.partition_id))
                continue;
            tile.updateFrequency(blockID2, ConfigParams.resSizeLimit, maxWindowSize);
        }
        affinities.put(blockID, tile);
    }

    public int getTileAccessCount(String tid) {
        if (affinities.containsKey(tid))
            return affinities.get(tid).accessCount;
        return 0;
    }

    public String getMostCoAccessedIDP(String tid, NewTileManager tileManager){
        String p = tileManager.getTilePartition(tid);
        return getMostCoAccessedIDP(tid,tileManager,p, new ArrayList<>());
    }

    public String getMostCoAccessedParForTile(String tid, NewTileManager tileManager){
        String p = tileManager.getTilePartition(tid);
        Map<String, Double> tileFreqs = affinities.get(tid).getFreqs();
        Map<String, Double> partitionFreqs = new HashMap<>();
        double maxFreq = 0;
        String maxP = "";
        for(Map.Entry<String, Double> tf:tileFreqs.entrySet()){
            String tfp = tileManager.getTilePartition(tf.getKey());
            if (!tfp.equals(p)) {
                if(partitionFreqs.containsKey(tfp))
                    partitionFreqs.merge(tfp, tf.getValue(), Double::sum);
                else
                    partitionFreqs.put(tfp, tf.getValue());

                if(partitionFreqs.get(tfp) > maxFreq){
                    maxFreq = partitionFreqs.get(tfp);
                    maxP = tfp;
                }
            }
        }
        return maxP;
    }

    public String getMostCoAccessedIDP(String tid, NewTileManager tileManager, String p, ArrayList<String> m) {
        Map<String, Double> tileFreqs = affinities.get(tid).getSortedFreqs();
        for(Map.Entry<String, Double> tf:tileFreqs.entrySet()){
            if (!tileManager.getTilePartition(tf.getKey()).equals(p) && !m.contains(tf.getKey())) {
                return tf.getKey();
            }
        }
        return "";
    }

    public double getAffinity(String tid, String tempID) {
        return affinities.get(tid).freqs.get(tempID);
    }

    public void multiplyWeights(double weightResetThreshold) {
        // TODO: test this
        for (Map.Entry<String, AffinityEntry> affEntry : affinities.entrySet()) {
            affEntry.getValue().freqs.replaceAll((k, v) -> v != null ? (weightResetThreshold * v) : null);
            affinities.put(affEntry.getKey(), affEntry.getValue());
        }
    }

    public static class AffinityEntry implements Serializable{
        public String bid;
        public Map<String, Double> freqs;
        public int accessCount;
        int decimalDigits = 6;

        public AffinityEntry(String bid, Map<String, Double> freqs, int freq_sum) {
            this.bid = bid;
            this.freqs = freqs;
            this.accessCount = freq_sum;
        }

        public AffinityEntry(String bid) {
            this.bid = bid;
            this.freqs = new HashMap<String, Double>();
            this.accessCount = 0;
        }

        public void updateFrequency(String bid, int totalTiles, int maxWindowSize) {
            double addedFreq = 1.0/(double) (maxWindowSize);
            // Just rounding :))
            addedFreq = Math.round(addedFreq * Math.pow(10, decimalDigits)) / Math.pow(10, decimalDigits);
            if(freqs.containsKey(bid)) {
                Double freq = freqs.get(bid);
                freq += addedFreq;
                freqs.put(bid, freq);
                return;
            }
            freqs.put(bid, addedFreq);
        }

        public void incrementAccessCount(){
            this.accessCount++;
        }

        @Override
        public String toString() {
            return
//                    "AffinityEntry{" +
//                    "bid='" + bid + '\'' +
                    " {freqs=" + freqs +
                    ", freq_sum=" + accessCount +
                    "}\n";
        }

        public Map<String, Double> getFreqs() {
            return this.freqs;
        }

        public Map<String, Double> getSortedFreqs() {
            freqs = UtilFUnctions.sortByValue(freqs, false);
            return freqs;
        }
    }
}
