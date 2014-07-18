package com.msl.naivebayes;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.net.UnknownHostException;
import java.util.Iterator;
import java.util.List;

import jvntagger.MaxentTagger;
import jvntagger.POSTagger;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import vn.hus.nlp.tokenizer.VietTokenizer;

import com.google.common.collect.HashMultiset;
import com.mongodb.BasicDBObject;
import com.mongodb.DB;
import com.mongodb.MongoClient;

public class NaiveBayes {
	POSTagger tagger;
	VietTokenizer tokenizer;
	public NaiveBayes() {
		tagger = new MaxentTagger("model/maxent");
		tokenizer = new VietTokenizer();
	}
	
	public void train(String[] fileName, int docsPerCls) {
		DB db = null;
		try {
			MongoClient mongoClient = new MongoClient("localhost", 27017);
			mongoClient.dropDatabase("trainingdb");
			db = mongoClient.getDB("trainingdb");
		} catch (UnknownHostException e) {
			e.printStackTrace();
		}
		
		int totalDocs = 0;
		JSONParser parser = new JSONParser();
		BasicDBObject[] doc = new BasicDBObject[fileName.length];
		for (int i = 0; i < fileName.length; i++) {
			HashMultiset<String> words = HashMultiset.create();
			String cls = fileName[i].substring(0, fileName[i].lastIndexOf("."));
			
			try {
				JSONArray clsData = (JSONArray) parser.parse(new FileReader("./data/" + 
						fileName[i]));
				System.out.println("Working on class: " + cls + "Size: " + clsData.size());
				//Selecting using fix number of doc per Class or not
				if(docsPerCls <= 0 || docsPerCls > clsData.size()) {
					docsPerCls = clsData.size();
				}
				
				totalDocs += docsPerCls;
				String url;
				String title;
				String content;
				for (int j = 0; j < docsPerCls; j++) {
					JSONObject jObj = (JSONObject) clsData.get(j);
					url = jObj.get("url").toString();
					title = jObj.get("title").toString();
					content = jObj.get("content").toString();
//					System.out.println(content);
//					content = title + ". " + content;
					content = content.replace("@", "").replace("\"", "").replace("'", "");
					String tokens = tokenizer.segment(content);
					String taggers = tagger.tagging(tokens.toLowerCase());
					String[] features = taggers.split(" ");
					for(int k=0; k<features.length; k++)
					{
						if((features[k].endsWith("/A") || features[k].endsWith("/V") || features[k].endsWith("/N")) && (!features[k].contains(".")))
							words.add(features[k]);
					}
				}
				
				doc[i] = new BasicDBObject();
				doc[i].append("cls", cls);
				doc[i].append("count", docsPerCls);
				doc[i].append("zero", (double) 1 / (words.elementSet().size() + words.size()));

				//Calculate features vector
				BasicDBObject features= new BasicDBObject();
				Iterator<String> elementIter = words.elementSet().iterator();
				while (elementIter.hasNext()) {
					String word = elementIter.next();
					double p = (double) (words.count(word) + 1)
							/ (words.elementSet().size() + words.size());
//					System.out.println(p);
					features.append(word, p);
				}
				doc[i].append("features", features);

//				System.out.println(words.elementSet().size() + " "
//						+ words.size());
//				System.out.println(dataForCls.size());

			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} catch (ParseException e) {
				e.printStackTrace();
			}
		}
		
		//Calculate the class's probability
		for(int i = 0; i<doc.length; i++)
		{
			double prop = (double) doc[i].getInt("count") / totalDocs;
			doc[i].append("prob", prop);
			db.getCollection(doc[i].getString("cls")).insert(doc[i]);
		}
		
	}
	
	public String classify(String fileName, int order, boolean showProb) {
		HashMultiset<String> words = HashMultiset.create();
		JSONParser parser = new JSONParser();
		try {
			JSONArray dataForCls = (JSONArray) parser.parse(new FileReader(fileName));
			if(order < dataForCls.size() && order >= 0) {
				
				JSONObject jObj = (JSONObject) dataForCls.get(order);
				String content = jObj.get("content").toString();
				content = content.replace("@", "").replace("\"", "").replace("'", "").replace(">>", "");
				if(showProb)
					System.out.println(content);
				String tokens = tokenizer.segment(content);
				String taggers = tagger.tagging(tokens.toLowerCase());
				String[] features = taggers.split(" ");
				for(int k=0; k<features.length; k++)
				{
					if((features[k].endsWith("/A") || features[k].endsWith("/V") || features[k].endsWith("/N")) && (!features[k].contains(".")))
						words.add(features[k]);
				}
			}
			
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		} catch (IOException e1) {
			e1.printStackTrace();
		} catch (ParseException e1) {
			e1.printStackTrace();
		}
		
		DB db = null;
		try {
			MongoClient mongoClient = new MongoClient("localhost", 27017);
			db = mongoClient.getDB("trainingdb");
		} catch (UnknownHostException e) {
			e.printStackTrace();
		}
		Iterator<String> iter = db.getCollectionNames().iterator();
		String selectedCategory = "";
		BigDecimal selectedP = new BigDecimal(-1);
		while(iter.hasNext())
		{
			String category = iter.next();
			if(!category.contains("system")) {
				Iterator<String> iterWords = words.elementSet().iterator();
				double zeroProp = (double) db.getCollection(category).distinct("zero").get(0);
				BigDecimal p = new BigDecimal((double) db.getCollection(category).distinct("prob").get(0));
				while(iterWords.hasNext()) {
					String word = iterWords.next();
					int count = words.count(word);
					List feature = db.getCollection(category).distinct("features."+word);
					
					if(feature.size() > 0) {
						p = p.multiply(new BigDecimal((double)feature.get(0) * count));
//						p *= (double)feature.get(0) * count;
					} else {
						p = p.multiply(new BigDecimal(zeroProp*count));
//						p *= zeroProp * count;
					}
//					System.out.println(p);
				}
				if(selectedP.compareTo(p) < 0) {
					selectedP = p;
					selectedCategory=category;		
				}
				if(showProb)
					System.out.format(category + ": %.5E\n", p);	
			}
		}
//		System.out.println("Doc belongs to " + selectedCategory + " with prob: " + selectedP);
		return selectedCategory;
	}
	
	public void evaluate(String cls, int start, int end, boolean showProb) {
		int wrong = 0;
		for(int i=start; i<end; i++) {
			String cat = classify("./data/" + cls + ".json", i, showProb);
			if(!cat.equals(cls)) {
				++wrong;
			}
		}
		int total = end - start;
		System.out.println("Total testcase: "+ total);
		System.out.println("Incorrect: " + wrong);
		System.out.println("Correct: " + (total-wrong) + " Ratio: " + (total-wrong)*100/total + "%");
	}
	
	public void getFeatures() {
		DB db = null;
		try {
			MongoClient mongoClient = new MongoClient("localhost", 27017);
			db = mongoClient.getDB("trainingdb");
		} catch (UnknownHostException e) {
			e.printStackTrace();
		}
		Iterator<String> iter = db.getCollectionNames().iterator();
		while(iter.hasNext())
		{
			String category = iter.next();
			if(!category.contains("system")) {
				List features = db.getCollection(category).distinct("features");
				System.out.println(category + " : " + ((BasicDBObject) features.get(0)).size());
			}
		}
	}
	
	public void classify(String path)
	{
		System.out.println("Start download page ...\n");
		String commands = "bash load_webpage.sh";
		try {
			Process p = Runtime.getRuntime().exec(commands);
			p.waitFor();
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}
		System.out.println("Download completed...\n");
		
		System.out.println("Result: Document belongs to category \""+ classify(path, 0, true) + "\"");
	}
	
	public static void main(String[] args) {
		if(args.length > 0) 
		{
			NaiveBayes naiveBayes = new NaiveBayes();
			
			int func = Integer.parseInt(args[0]);
			switch (func) {
			case 0:
				naiveBayes.train(new File("./data").list(), Integer.parseInt(args[1]));
				break;
			case 1:
				naiveBayes.evaluate(args[1], Integer.parseInt(args[2]), Integer.parseInt(args[3]), Integer.parseInt(args[4]) == 0? false: true);
				break;
			case 2:
				naiveBayes.classify("./page/page.json");
				break;
			}
		} else {
			System.out.println("Please select functions()");
		}
		
	}

}
