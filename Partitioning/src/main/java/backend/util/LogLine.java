package backend.util;

import java.io.Serializable;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;

public class LogLine implements Serializable {
    public String seqNUm;
    public Date theTime;
    public String clientIP;
    public int resSize;
    public QueryInfo query;

    public LogLine() {
        this.theTime = null;
        this.query = new QueryInfo("stmt");
    }
    public LogLine(Date theTime, String clientIP, int resSize, String sqlStatement) {
        this.theTime = theTime;
        this.clientIP = clientIP;
        this.resSize = resSize;
        this.query = new QueryInfo(sqlStatement);
    }

    public LogLine(Date theTime, String clientIP, int resSize, String sqlStatement, ArrayList<String> resultSet) {
        this.seqNUm = String.valueOf(1);
        this.theTime = theTime;
        this.clientIP = clientIP;
        this.resSize = resSize;
        this.query = new QueryInfo(sqlStatement, resultSet);
    }

    public LogLine(int count, String[] columns) {
        DateFormat dateFormat = new SimpleDateFormat("MM/dd/yyyy hh:mm:ss a");
        try {
            this.seqNUm = String.valueOf(count);
            this.theTime = dateFormat.parse(columns[0]);
            this.clientIP = columns[1];
            this.resSize = Integer.parseInt(columns[2]);
            if(columns.length > 4)
                this.query = new QueryInfo(columns[3], columns[4]);
            else
                this.query = new QueryInfo(columns[3]);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            System.out.println("execption while parsing " + count + " " + columns[0]);
        }
    }

    public void setResSize(int resSize) {
        this.resSize = resSize;
    }

    @Override
    public String toString() {
        return "\nLogLine{" +
                "theTime=" + theTime +
                ", clientIP='" + clientIP + '\'' +
                ", resSize=" + resSize +
                ", sqlStatement='" + query.statement + '\'' +
                ", SqlRes='" + query.result_set + '\'' +
                '}';
    }

    public boolean isInSameSession(LogLine log) {
        if(!log.clientIP.equals(this.clientIP)) {
            return false;
        }
        if(Math.abs(this.theTime.getTime() - log.theTime.getTime()) > 1800000) {
            return false;
        }

        return true;
    }

    public String getSummerisedString() {
        String pattern = "MM/dd/yyyy hh:mm:ss a";
        DateFormat df = new SimpleDateFormat(pattern);
        return df.format(theTime) + "||" + clientIP + "||" + resSize + "||" + query.statement + "\n";
    }
}
