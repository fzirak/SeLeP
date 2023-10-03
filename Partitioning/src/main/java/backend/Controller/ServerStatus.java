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

@WebServlet("/ready")
public class ServerStatus extends HttpServlet {
    public void init(){}

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        System.out.println("in status do get");

        System.out.println("--- check server status.");

        String res = (ServerMainThread.isReadyToTest()) ? "1" : "0";

        response.setContentType("text/html");
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().println(res);
    }
}
