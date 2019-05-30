package librec.data;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by jiaxit
 */

public class ArffAttribute {
    private static final Set<String> VALID_TYPES = new HashSet<>(Arrays.asList(
            new String[] {"NUMERIC", "REAL", "INTEGER", "STRING", "NOMINAL"}
    ));

    private String name;
    private String type;
    private int idx;

    private Set<String> columnSet;

    public ArffAttribute(String name, String type, int idx){
        // check if type is valid
        if (!VALID_TYPES.contains(type)) {
            throw new IllegalArgumentException("Invalid Type: " + type);
        }

        this.name = name;
        this.type = type;
        this.idx = idx;
    }

    public String getName() {
        return name;
    }

    public String getType() {
        return type;
    }

    public int getIndex() {
        return idx;
    }

    public Set<String> getColumnSet() {
        return columnSet;
    }

    public void setColumnSet(Set<String> columnSet) {
        this.columnSet = columnSet;
    }

}
