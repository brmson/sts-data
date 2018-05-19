                       SemEval 2017 Shared Task 1
                    Semantic Textual Similarity (STS)

This package contains the test sets for the 2017 Semantic Textual Similarity
(STS) shared task. Each evaluation set has the following tab-separated format:

  * One STS pair per line.
  * Each line contains the following fields: STS Sent1, STS Sent2

The input files are provided in UTF-8.

Example:

Un perro salta sobre los obstáculos para un show canino.        A dog jumps over something.

One file is provided for each track of STS 2017, with the exception of track 4.
Track 4 is subdivided into two subtracks, one for data drawn from SNLI and
another from data sourced from WMT news data. All other datasets were sourced from
SNLI.

Input Files (UTF-8):
--------------------
STS.input.track1.ar-ar.txt
STS.input.track2.ar-en.txt
STS.input.track3.es-es.txt
STS.input.track4a.es-en.txt
STS.input.track4b.es-en.txt
STS.input.track5.en-en.txt
STS.input.track6.tr-en.txt

Output Files: 
-------------
For each evaluation set, please generate a plain text output file with one line
for each STS pair that provides the score assigned by your system as a floating
point number:

0.1
4.9
3.5
2.0
5.1

Release History
---------------
v1.0 Jan 16, 2017 - Initial release of the test set
v1.1 Jan 18, 2017 - Correction to track 1, Arabic-Arabic STS 

Output files should be uploaded to the STS 2017 CodaLab site:

https://competitions.codalab.org/competitions/16051

Each team can submit up to three runs. 

Good Luck!
