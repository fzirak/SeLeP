package backend.util;

import java.util.ArrayList;
import java.util.HashMap;

public class DBComponents {
    public HashMap<String, DBObject> components;

    public DBComponents() {
        components = new HashMap<>();
    }

    public void setComponents(HashMap<String, DBObject> components) {
        this.components = components;
    }

    public HashMap<String, DBObject> getComponents() {
        return components;
    }

    public void addComponent(DBObject new_comp){
        this.components.put(new_comp.name, new_comp);
    }

    public DBObject getComponent(String name){
        return this.components.get(name);
    }

    @Override
    public String toString() {
        return "DBComponents{" +
                "components=" + components +
                '}';
    }
}
