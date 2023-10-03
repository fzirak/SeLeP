package backend.disk;

import backend.util.NewTile;

import java.util.HashMap;
import java.util.Map;

public class NewTileManager {
    public Map<String, NewTile> tiles;

    public NewTileManager() {
        tiles = new HashMap<>();
    }

    public NewTileManager(Map<String, NewTile> tiles) {
        this.tiles = tiles;
    }

    public NewTile getTile(String tid){
        return tiles.get(tid);
    }

    public void addTile(NewTile tile){
        tiles.put(tile.tid, tile);
    }

    public String getTilePartition(String tid){
        return tiles.get(tid).partition_id;
    }

    public void setTilePartition(String tid, String candidatePartition) {
        tiles.get(tid).setPartition_id(candidatePartition);
    }
}
