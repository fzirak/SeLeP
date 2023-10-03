package backend.util;

public class NewTile {
    public String tid;
    public String partition_id;

    public NewTile(String tid, String partition_id) {
        this.tid = tid;
        this.partition_id = partition_id;
    }

    public NewTile(String tid) {
        this.tid = tid;
        this.partition_id = "";
    }

    public void setPartition_id(String partition_id) {
        this.partition_id = partition_id;
    }

    @Override
    public String toString() {
        return "NewTile{" +
                "tid='" + tid + '\'' +
                ", partition_id='" + partition_id + '\'' +
                '}';
    }
}
