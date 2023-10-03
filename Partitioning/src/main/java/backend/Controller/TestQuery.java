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

@WebServlet("/test")
public class TestQuery extends HttpServlet {
    public void init(){}

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        System.out.println("in test do post");

        LogLine q = new Gson().fromJson(request.getParameter("query"), LogLine.class);
        System.out.println("--- New Test Query: " + q.query.statement +
                "\nTotal " + q.query.result_set.size() + " tiles.");
        ServerMainThread.processNewTestQuery(q);
        int resCount = DatabaseUtil.executePSQLQuery(q.query.statement);

        response.setContentType("text/html");
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().println(String.format("successfully retrieved %d objects\n", resCount));
    }
}
