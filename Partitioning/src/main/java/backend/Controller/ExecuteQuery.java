package backend.Controller;

import backend.Database.DatabaseUtil;
import backend.ServerMainThread;
import backend.util.LogLine;
import com.google.gson.Gson;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;

import static Frontend.Client.partition_initializer_workload;

@WebServlet("/exec")
public class ExecuteQuery extends HttpServlet {
    public void init(){}
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        System.out.println("get request in exec query");
        System.out.println(request);

        LogLine q = new Gson().fromJson(request.getParameter("query"), LogLine.class);
        System.out.println("--- New Query: " + q.query.statement + "\n    Accessed tiles: " + q.query.result_set);
        ServerMainThread.processNewQuery(q);
//        System.out.println(q.query.statement);
        int resCount = DatabaseUtil.executePSQLQuery(q.query.statement);

        response.setContentType("text/html");
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().println(String.format("successfully retrieved %d objects\n", resCount));
        System.out.println("response is sent");
    }

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        System.out.println("in do post");
        LogLine q = new Gson().fromJson(request.getParameter("query"), LogLine.class);
        String alterParam = request.getParameter("alter");
        if (alterParam.equals("1")) {
            String queryIdx = request.getParameter("queryidx");
            if (queryIdx != null) {
                System.out.println("--- Reading Query " + queryIdx);
                ServerMainThread.readQueryLocally(q, queryIdx);
            }
            else{
                System.out.println("--- New Query: " + q.query.statement +
                        //                "\n    Accessed tiles: " + q.query.result_set +
                        "\nTotal " + q.query.result_set.size() + " tiles.");
                ServerMainThread.processNewQuery(q);
            }
        }
        else{
            String fileName = request.getParameter("fname");
            System.out.println(fileName);
            ServerMainThread.addResPartition(q, fileName);
        }

//        System.out.println(q.query.statement);
//        int resCount = DatabaseUtil.executePSQLQuery(q.query.statement);
        int resCount = 1;
        response.setContentType("text/html");
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().println(String.format("successfully retrieved %d objects\n", resCount));
    }
}
