package backend.util;

import java.io.Serializable;

public enum DBObjectType implements Serializable {
    table,
    view,
    function,
    stored_procedure,
    unknown;

    public static DBObjectType getType(String s) {
        if (s.contains("table"))
            return table;
        else if (s.contains("view"))
            return view;
        else if (s.contains("function"))
            return function;
        else if (s.contains("procedure"))
            return stored_procedure;
        else return unknown;
    }
}
