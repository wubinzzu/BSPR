package librec.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;

import com.google.common.collect.*;
import librec.util.FileIO;

public class ArffDataLoader {
	public String dataPath;
    public String relationName;
    protected ArrayList<ArffInstance> instances;
    protected ArrayList<ArffAttribute> attributes;
    protected ArrayList<String> attrTypes;
    private ArrayList<BiMap<String, Integer>> columnIds;
    private int ratingCol;

    public SparseMatrix oneHotFeatureMatrix;
    public DenseVector oneHotRatingVector;

	public ArffDataLoader(String path){
		dataPath = path;
        instances = new ArrayList<>();
        attributes = new ArrayList<>();
        columnIds = new ArrayList<>();
        attrTypes = new ArrayList<>();

        ratingCol = -1;
	}
	
	public void readData() throws IOException{
		BufferedReader br = FileIO.getReader(dataPath);
		boolean dataFlag = false;
		
		int attrIdx = 0;

		String attrName = null;
		String attrType = null;
		String line = null;

		while (true) {

            // parse DATA if valid
            if (dataFlag) {
                // let data reader control the bufferedReader
                dataReader(br);
            }

            line = br.readLine();

            if (line == null) // finish reading
                break;
            if (line.isEmpty() || line.startsWith("%")) // skip empty or annotation
                continue;

            String[] data = line.trim().split("[ \t]");

            // parse RELATION
            if (data[0].toUpperCase().equals("@RELATION")) {
                relationName = data[1];
            }

            // parse ATTRIBUTE
            else if (data[0].toUpperCase().equals("@ATTRIBUTE")) {
                attrName = data[1];
                attrType = data[2];
                boolean isNominal = false;

                if (attrName.equals("user")) {
				}
                if (attrName.equals("item")) {
				}
                if (attrName.equals("rating"))
                    ratingCol = attrIdx;

                // parse NOMINAL type
                if (attrType.startsWith("{") && attrType.endsWith("}")) {
                    isNominal = true;
                }

                BiMap<String, Integer> colId = HashBiMap.create();
                // if nominal type, set columnIds
                if (isNominal) {
                    String nominalAttrs = attrType.substring(1,attrType.length() - 1);
                    int val = 0;
                    for (String attr: nominalAttrs.split(",")) {
                        colId.put(attr.trim(), val ++);
                    }
                    attrType = "NOMINAL";
                }

                columnIds.add(colId);
                attributes.add(new ArffAttribute(attrName, attrType.toUpperCase(), attrIdx++));
            }

            // set DATA flag (finish reading ATTRIBUTES)
            else if (data[0].toUpperCase().equals("@DATA")){
                dataFlag = true;
            }
        }

        // initialize attributes
        for (int i = 0; i < attributes.size(); i ++ ) {
            attributes.get(i).setColumnSet(columnIds.get(i).keySet());
        }
        // initialize instance attributes
        ArffInstance.attrs = attributes;
	}

    private void dataReader(Reader rd) throws IOException {
        ArrayList<String> dataLine = new ArrayList<>();
        StringBuilder subString = new StringBuilder();
        boolean isInQuote = false;
        boolean isInBracket = false;

        // get all attribute types
        for (ArffAttribute attr: attributes) {
            attrTypes.add(attr.getType());
        }

        int c = 0;
        while ((c = rd.read()) != -1) {
            char ch = (char) c;

            // read line by line
            if (ch == '\n'){
                if (dataLine.size() != 0) {  // check if empty line
                    if (!dataLine.get(0).startsWith("%")) {  // check if annotation line
                        dataLine.add(subString.toString());
                        // raise error if inconsistent with attribute define
                        if (dataLine.size() != attrTypes.size()) {
                            throw new IOException("Read data error, inconsistent attribute number!");
                        }

                        // pul column value into columnIds, for one-hot encoding
                        for (int i = 0; i < dataLine.size(); i ++ ) {
                            String col = dataLine.get(i).trim();
                            String type = attrTypes.get(i);
                            BiMap<String, Integer> colId = columnIds.get(i);
                            switch (type) {
                                case "NUMERIC":
                                case "REAL":
                                case "INTEGER":
                                    break;
                                case "STRING":
                                    int val = colId.containsKey(col) ? colId.get(col) : colId.size();
                                    colId.put(col, val);
                                    break;
                                case "NOMINAL":
                                    StringBuilder sb = new StringBuilder();
                                    String[] ss = col.split(",");
                                    for (int ns = 0; ns < ss.length; ns ++ ) {
                                        String _s = ss[ns].trim();
                                        if (!colId.containsKey(_s)) {
                                            throw new IOException("Read data error, inconsistent nominal value!");
                                        }
                                        sb.append(_s);
                                        if (ns != ss.length - 1)
                                            sb.append(",");
                                    }
                                    col = sb.toString();
                                    break;
                            }
                            dataLine.set(i, col);
                        }

                        instances.add(new ArffInstance(dataLine));

                        subString = new StringBuilder();
                        dataLine = new ArrayList<>();
                    }
                }
            } else if (ch == '[' || ch == ']') {
                isInBracket = !isInBracket;
            } else if (ch == '\r') {
                // skip '\r'
            } else if (ch == '\"') {
                isInQuote = !isInQuote;
            } else if (ch == ',' && (!isInQuote && !isInBracket)) {
                dataLine.add(subString.toString());
                subString = new StringBuilder();
            } else {
                subString.append(ch);
            }
        }
    }

    @SuppressWarnings("unchecked")
	public void oneHotEncoding() {
        Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
        Multimap<Integer, Integer> colMap = HashMultimap.create();

        int numRows = instances.size();
        int numCols = 0;
        int numAttrs = attributes.size();

        double[] ratings = new double[numRows];

        // set numCols
        for (int i = 0; i < attributes.size(); i ++ ) {
            // skip rating column
            if (i == ratingCol)
                continue;

            ArffAttribute attr = attributes.get(i);
            numCols += attr.getColumnSet().size() == 0 ? 1 : attr.getColumnSet().size();
        }

        // build one-hot encoding matrix
        for (int row = 0; row < numRows; row ++ ) {
            ArffInstance instance = instances.get(row);
            int colPrefix = 0;
            int col = 0;
            for (int i = 0; i < numAttrs; i ++ ) {
                String type = attrTypes.get(i);
                Object val = instance.getValueByIndex(i);

                // rating column
                if (i == ratingCol) {
                    ratings[row] = (double) val;
                    continue;
                }

                // feature column
                switch (type) {
                    case "NUMERIC":
                    case "REAL":
                    case "INTEGER":
                        col = colPrefix;
                        dataTable.put(row, col, (double) val);
                        colMap.put(col, row);
                        colPrefix += 1;
                        break;
                    case "STRING":
                        col = colPrefix + columnIds.get(i).get(val);
                        dataTable.put(row, col, 1d);
                        colMap.put(col, row);
                        colPrefix += columnIds.get(i).size();
                        break;
                    case "NOMINAL":
                        for (String v: (ArrayList<String>) val) {
                            col = colPrefix + columnIds.get(i).get(v);
                            colMap.put(col, row);
                            dataTable.put(row, col, 1d);
                        }
                        colPrefix += columnIds.get(i).size();
                        break;
                }
            }
        }
        oneHotFeatureMatrix = new SparseMatrix(numRows, numCols, dataTable, colMap);
        oneHotRatingVector = new DenseVector(ratings);

        // release memory
        dataTable = null;
        colMap = null;
    }

    public String getRelationName() {
        return relationName;
    }

    public ArrayList<ArffInstance> getInstances() {
        return instances;
    }

    public ArrayList<ArffAttribute> getAttributes() {
        return attributes;
    }
}
