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

@WebServlet("/serverdata")
public class ServerData extends HttpServlet {
    public void init(){}

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        System.out.println("get request in server data controller");
        String serverDataName = request.getParameter("name");
        String fileName = request.getParameter("filename");
        if (fileName == null)
            fileName = serverDataName;
        System.out.println("get request for " + serverDataName);
        ServerMainThread.returnServerInfo(serverDataName, fileName);
        response.setContentType("text/html");
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().println("successfully retrieved server data\n");
        System.out.println("response is sent");
    }

}
