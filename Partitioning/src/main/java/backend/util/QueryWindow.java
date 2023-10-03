package backend.util;

import java.util.LinkedList;
import java.util.Queue;

public class QueryWindow {
    public Queue<LogLine> queryWindow;
    public int windowSize;

    public QueryWindow(int windowSize) {
        this.windowSize = windowSize;
        this.queryWindow = new LinkedList<>();
    }

    public void insertQuery(LogLine logline){
        if(queryWindow.size() >= windowSize){
            queryWindow.poll();
            queryWindow.add(logline);
        }
        queryWindow.add(logline);
    }

    public void clearQueryWindow(){
        queryWindow.clear();
    }

    public int getMaxQueryWindowSize(){
        return queryWindow.size();
    }

    public int getSize(){
        return queryWindow.size();
    }

    @Override
    public String toString() {
        return "QueryWindow{" +
                "queryWindow=" + queryWindow +
                ", windowSize=" + windowSize +
                '}';
    }
}
