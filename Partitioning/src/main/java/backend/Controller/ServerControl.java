package backend.Controller;

import backend.ServerMainThread;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@WebServlet("/servercntl")
public class ServerControl extends HttpServlet {
    public void init(){}

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String cntlCommand = request.getParameter("command");
        if (cntlCommand.equals("stop")){
            ServerMainThread.stopServer();
        }
        else if (cntlCommand.equals("clearwindow")){
            ServerMainThread.clearQueryWindow();
        }
        response.setContentType("text/html");
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().println("1");
    }
}
