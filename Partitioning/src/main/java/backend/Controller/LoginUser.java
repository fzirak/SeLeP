package backend.Controller;

import backend.ServerMainThread;
import backend.util.LogLine;
import com.google.gson.Gson;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.net.URLDecoder;
import java.net.URLEncoder;

@WebServlet("/login")
public class LoginUser extends HttpServlet {
    public void init(){}
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        System.out.println("get request in login");
        System.out.println(request);
        LogLine q = new Gson().fromJson(request.getParameter("query"), LogLine.class);
        response.setContentType("text/html");
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().println("Hello new client!\n");

    }
}
