/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mlssdd.github.git;

import java.io.File;
import org.eclipse.jgit.api.Git;

/**
 *
 * @author mosesopenja
 */
public class CloneRepository {

    public static String cloneR(String repos, String path) {
        try {
            System.out.println("     ---- cloning " + repos + " ... please wait ...");
            File file = new File(path + repos.split("/")[1]);
            if (file.exists()) {
                return path + repos.split("/")[1];
            } else {
                Git git = Git.cloneRepository()
                        .setURI("https://github.com/" + repos + ".git")
                        .setDirectory(file)
                        .call();
                System.out.println(" ---- cloning done successfull!!");
                return path + repos.split("/")[1];
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println(" ---- Error: while cloning.. ");
            return "";
        }
    }
}
