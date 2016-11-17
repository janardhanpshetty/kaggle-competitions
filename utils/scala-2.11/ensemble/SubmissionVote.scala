package ensemble.ml

import java.io._
import scala.io.Source

// important for 'foreach'
/**
 * Created by jshetty on 8/23/16.
 * This code takes directory name with all csv submission files(id, prediction) and write final submission file
 * Usage:scalac SubmissionVote.scala /Users/jshetty/spark-applications/kaggle/output/redhat/result /Users/jshetty/spark-applications/kaggle/output/redhat/result/combFile activity_id outcome
 */
object SubmissionVote {

  def getListOfFiles(dir: File, extensions: List[String]): List[File] = {
    dir.listFiles.filter(_.isFile).toList.filter { file =>
      extensions.exists(file.getName.endsWith(_))
    }
  }

  def topOnCount(inputList: List[Double]): Double = {
    val mostFreqItem = inputList.groupBy(identity).maxBy(_._2.size)._1
    mostFreqItem
  }

  def main(args: Array[String]): Unit = {

    if (args.length < 4) {
      System.err.println("Usage: <directory> <output_file> <field1> <field2>")
      System.exit(1)
    }
    val directory = args(0)
    val outFile = args(1)
    val field1 = args(2)
    val field2 = args(3)

    /*val directory = "/Users/jshetty/spark-applications/kaggle/output/redhat/result"
    val outFile = "/Users/jshetty/spark-applications/kaggle/output/redhat/result/combFile"
    val field1 = "activity_id"
    val field2 = "outcome"*/
    val submission = collection.mutable.Map[String, List[Double]]()
    val fileExtensions = List("csv")
    val files = getListOfFiles(new File(directory), fileExtensions)

    val keys =files.flatMap(fileName => Source.fromFile(fileName).getLines.toList
      .map(line => (line.split(",")(0),line.split(",")(1))
      ))

    // Add to the Hashmap
    keys.foreach(line =>
      submission.get(line._1) match {
        case None => submission(line._1) = List(line._2.toDouble)
        case Some(value) =>  submission(line._1) = submission(line._1) ++ List(line._2.toDouble)
      }
    )
    // Get the top 1 item based on count
    val finalSubmission = submission.mapValues(x => topOnCount(x))
    // Take each element and take a major
    val file = new File(outFile)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(field1 + "," + field2)
    bw.newLine()
    // Write Hashmap into a file line by line
    finalSubmission.foreach{
      line => bw.write(line.toString().substring(1, line.toString().length - 1))
        bw.newLine()
    }

    bw.close()
  }
}
