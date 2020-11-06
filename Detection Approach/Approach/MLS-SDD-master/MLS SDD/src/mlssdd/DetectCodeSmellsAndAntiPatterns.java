/* (c) Copyright 2019 and following years, MounaA and PalmyreB.
 *
 * Use and copying of this software and preparation of derivative works
 * based upon this software are permitted. Any copy of this software or
 * of any derivative work must include the above copyright notice of
 * the author, this paragraph and the one after it.
 *
 * This software is made available AS IS, and THE AUTHOR DISCLAIMS
 * ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE, AND NOT WITHSTANDING ANY OTHER PROVISION CONTAINED HEREIN,
 * ANY LIABILITY FOR DAMAGES RESULTING FROM THE SOFTWARE OR ITS USE IS
 * EXPRESSLY DISCLAIMED, WHETHER ARISING IN CONTRACT, TORT (INCLUDING
 * NEGLIGENCE) OR STRICT LIABILITY, EVEN IF THE AUTHOR IS ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGES.
 *
 * All Rights Reserved.
 */

package mlssdd;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;
import mlssdd.core.Directory;

import mlssdd.codesmells.detection.repository.UnusedImplementationDetection;

import mlssdd.antipatterns.detection.IAntiPatternDetection;
import mlssdd.antipatterns.detection.repository.ExcessiveInterLanguageCommunicationDetection;
import mlssdd.antipatterns.detection.repository.TooMuchClusteringDetection;
import mlssdd.antipatterns.detection.repository.TooMuchScatteringDetection;
import mlssdd.codesmells.detection.ICodeSmellDetection;
import mlssdd.codesmells.detection.repository.AssumingSafeMultiLanguageReturnValuesDetection;
import mlssdd.codesmells.detection.repository.HardCodingLibrariesDetection;
import mlssdd.codesmells.detection.repository.LocalReferencesAbuseDetection;
import mlssdd.codesmells.detection.repository.MemoryManagementMismatchDetection;
import mlssdd.codesmells.detection.repository.NotCachingObjectsElementsDetection;
import mlssdd.codesmells.detection.repository.NotHandlingExceptionsDetection;
import mlssdd.codesmells.detection.repository.NotSecuringLibrariesDetection;
import mlssdd.codesmells.detection.repository.NotUsingRelativePathDetection;
import mlssdd.codesmells.detection.repository.PassingExcessiveObjectsDetection;
import mlssdd.codesmells.detection.repository.UnusedDeclarationDetection;
import mlssdd.codesmells.detection.repository.UnusedParametersDetection;
import mlssdd.utils.CodeToXml;

public class DetectCodeSmellsAndAntiPatterns {

	/**
	 * Outputs a CSV listing the code smells and anti-patterns detected in the input
	 * project.
	 *
	 * @param args Path to the input project (can be a file or a directory)
	 */
	private static final String PATH = "D:/Mona/new-downloads/";
	private static final String[] PROJECTS = { "vlc" };

	public static void main(String[] argumnets) {
		System.out.println("starting :  ");
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		

		for (int a = 0; a < PROJECTS.length; a++) {
			// String project = CloneRepository.cloneR(PROJECTS[a], PATH);
			String projectPath = PATH + PROJECTS[a] + "_snapshots/";
			List<String> list = Directory.getFolder(projectPath);
			
			//========start====
			final Set<ICodeSmellDetection> codeSmellDetectors = new HashSet<>();
			final Set<IAntiPatternDetection> antiPatternDetectors = new HashSet<>();

			codeSmellDetectors.add(new AssumingSafeMultiLanguageReturnValuesDetection());
			codeSmellDetectors.add(new HardCodingLibrariesDetection());
			codeSmellDetectors.add(new LocalReferencesAbuseDetection());
			codeSmellDetectors.add(new MemoryManagementMismatchDetection());
			codeSmellDetectors.add(new NotHandlingExceptionsDetection());
			codeSmellDetectors.add(new NotSecuringLibrariesDetection());
			codeSmellDetectors.add(new NotUsingRelativePathDetection());
			codeSmellDetectors.add(new PassingExcessiveObjectsDetection());
			codeSmellDetectors.add(new UnusedParametersDetection());

			// Detectors that need to analyse both languages
			// Uncomment when giving both Java and native code as an argument
			codeSmellDetectors.add(new NotCachingObjectsElementsDetection());
			codeSmellDetectors.add(new UnusedDeclarationDetection());
			codeSmellDetectors.add(new UnusedImplementationDetection());

			antiPatternDetectors.add(new ExcessiveInterLanguageCommunicationDetection());
			antiPatternDetectors.add(new TooMuchClusteringDetection());
			antiPatternDetectors.add(new TooMuchScatteringDetection());
			
			//========end=========
			
			for (int b = 73; b < list.size(); b++) { // 
				Calendar cal = Calendar.getInstance();
				System.out.println("Time: "+ dateFormat.format(cal.getTime()));
//				String project = list.get(b);
				String project = projectPath+PROJECTS[a]+"_"+b;
				System.out.println("Detecting smells for "+ project);
				// ArrayList< Object[]> allobj = new ArrayList<Object[]>();
				//System.exit(1);
				// public static void main(final String[]) {
				
				final long start = System.currentTimeMillis();

				final Document xml = CodeToXml.parse(project);
				System.out.println("The creation of the XML took " + (System.currentTimeMillis() - start) + " ms.\n");
//				System.exit(1);
				try {
					int id = 0;
					String bareName = "";
					if (bareName.equals("")) {
						//final String[] parts = argumnets[0].split("[\\/\\\\]");
						final String[] parts = project.split("[\\/\\\\]");
						bareName = parts[parts.length - 1];
					}
					final String dir = "results";
					final String fullPath = dir + "/" + bareName + ".csv";
					System.out.println("Result path: "+fullPath);
//					final String xmlPath = dir + "/" + bareName + ".xml";
					
//					try {
//					TransformerFactory transformerFactory = TransformerFactory.newInstance();
//					Transformer transformer = transformerFactory.newTransformer();
//					
//					DOMSource source = new DOMSource(xml);
//					FileWriter writer = new FileWriter(new File(xmlPath));
//					StreamResult result = new StreamResult(writer);
//					
//					transformer.transform(source, result);
//					System.out.println(xmlPath);
//					
//					Transformer tr = TransformerFactory.newInstance().newTransformer();
//		            tr.setOutputProperty(OutputKeys.INDENT, "yes");
//		            tr.setOutputProperty(OutputKeys.METHOD, "xml");
//		            tr.setOutputProperty(OutputKeys.ENCODING, "UTF-8");
//		            tr.setOutputProperty(OutputKeys.DOCTYPE_SYSTEM, "roles.dtd");
//		            tr.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
//
//		            // send DOM to file
//		            tr.transform(new DOMSource(xml), 
//		                                 new StreamResult(new FileOutputStream(xmlPath)));
//					System.exit(1);
					
//					}catch(Exception e) {
//						System.out.println(e.toString());
//					}

					if (new File(dir).mkdirs()) {
						System.out.println("Directory " + dir + " created");
					}

//					System.out.println(bareName);
					//System.out.println(argumnets[0]);
//					System.out.println(project);
//					System.out.println();

					// FileWriter(..., false): no auto-append, write at the beginning of the file
					// PrintWriter(..., false): no autoflush for performance reason
					final PrintWriter outputWriter = new PrintWriter(
							new BufferedWriter(new FileWriter(fullPath, false)), false);
					outputWriter.println("ID,Name,Variable,Method,Class,Package,File,File Name");

					for (final ICodeSmellDetection detector : codeSmellDetectors) {
						System.out.println("Detector: "+detector.toString());
						detector.detect(xml);
//						System.out.println("Saving results...");
						detector.output(outputWriter, id);
						final int nbCodeSmells = detector.getCodeSmells().size();
						id += nbCodeSmells;
						System.out.println(detector.getCodeSmellName() + ": " + nbCodeSmells);
					}

					for (final IAntiPatternDetection detector : antiPatternDetectors) {
						System.out.println("Detector: "+detector.toString());
						detector.detect(xml);
						detector.output(outputWriter, id);
						final int nbAntiPatterns = detector.getAntiPatterns().size();
						id += nbAntiPatterns;
						System.out.println(detector.getAntiPatternName() + ": " + nbAntiPatterns);
					}
					outputWriter.flush();
					outputWriter.close();
					System.out.println("\nThe detection took " + (System.currentTimeMillis() - start) + " ms.");
					cal = Calendar.getInstance();
					System.out.println("\nThe detection completed at: " + dateFormat.format(cal.getTime()));
				} catch (final IOException e) {
					System.out.println("Cannot create output file");
					e.printStackTrace();
				}
//				break;
			}
		}
	}

}