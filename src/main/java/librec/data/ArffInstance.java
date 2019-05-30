package librec.data;

import java.util.ArrayList;
import java.util.Arrays;

public class ArffInstance {
    public static ArrayList<ArffAttribute> attrs;
    private ArrayList<String> instanceData;

	public ArffInstance(ArrayList<String> instanceData) {
        this.instanceData = instanceData;
	}
	
	public Object getValueByAttrName(String attrName) throws Exception{
        Object res = null;
        boolean isNameValid = false;
        for (ArffAttribute attr: attrs) {
            if (attrName.equals(attr.getName())) {
                res = getValueByIndex(attr.getIndex());
                isNameValid = true;
                break;
            }
        }
        if (isNameValid == false)
            throw new Exception("invalid attrName: " + attrName);
        return res;
	}

    public Object getValueByIndex(int idx) {
        Object res = new Object();
        switch (getTypeByIndex(idx).toUpperCase()) {
            case "NUMERIC":case "REAL":case "INTEGER":
                res = Double.parseDouble(instanceData.get(idx));
                break;
            case "STRING":
                res = instanceData.get(idx);
                break;
            case "NOMINAL":
                String data[] = instanceData.get(idx).split(",");
                res = new ArrayList<>(Arrays.asList(data));
                break;
        }
        return res;
    }

    public String getTypeByIndex(int idx) {
        ArffAttribute attr = attrs.get(idx);
        return attr.getType();
    }
}
