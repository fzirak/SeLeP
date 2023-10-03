package backend.util;

import java.util.ArrayList;

public class Clump {
    public ArrayList<String> tiles;
    public String candidatePartition;

    public Clump() {
        tiles = new ArrayList<>();
    }

    public boolean isEmpty() {
        return tiles.size() == 0;
    }

    public void setCandidatePartition(String candidatePartition) {
        this.candidatePartition = candidatePartition;
    }

    public void setTiles(ArrayList<String> tiles) {
        this.tiles = tiles;
    }

    public void addTile(String tid){
        this.tiles.add(tid);
    }

    public void clear() {
        tiles.clear();
        candidatePartition = "";
    }
}
