# API checkers

This toolkit is to perform API update, and JS object conversion based on the HTML text contents.

## API HTML page download and API compare & update

Execute 'Compare\html_download.py' to download **English API HTML** files of expected versions to specified directory (you can input the 
version and directory after "python html_download.py")

- Some tips:	
	- **ALERT** : tested in python 2.7.15, the package ``urllib2`` imported may not be compatible with python 3
	- Make sure the Internet connection works well before downloading.

- To compare API 1.3 and develop (newest API), we run the script twice, the first time to download the *1.3 API* , 
  and the second time for downloading *develop API*. The best practice is to download 1.3 to a folder named 'old' and develop
  to 'new' .

- Move the two folders to ``Compare`` if they are not there. Then open template-new.html and template-old.html in a browser. Simply select all and copy their contents to some compare software and compare.  Done.

Instead of comparing the HTML pages fully rendered, this method is much more effective for reasonable formatting is maintained so that the API 
components are distinguishable enough for translators. You may finish compare two versions of API in less than 5 minutes.

We only need to compare en API to translate.



## Argument Checker --- CN

Sometimes, the argument explanation part may not match the arguments in the blue block. 
For example, in blue block, the argument is paddle.fluid.someone(A,B,C,D) however 
in argument explanation it only explains A,B,D. C is missed.
To pick out APIs with this flaw, CN_argument_Checker is developed.

CN API is hand-written so it is much easier to extract the two sections of arguments and compare them.




## Analyze API Example Code + API Excel Generation

### API ==> JS object and display them in <table>

**ALERT: Please move the API HTML you downloaded to API_to_Table/api_en_html**

Simply run ``API_to_Table/en_apis.html`` in ``Firefox`` (for Chrome DOESN'T allow non-http protocol to access blob contents) .

**BUG TO FIX** : the latest Firefox disabled multiple downloads from a single web page. 



### API <table> ==> Excel table 

1. Display all the API in the web page, then in the browser, inspect element => copy the HTML code to another HTML file
2. From Excel, from the upper control panel, choose Data ==> From HTML , done.
3. Merge cells according to the first column:

		```
		Sub Demo()
	    Dim Rng As Range
	    Dim TempStr As String
	    Dim Cell As Range
	 
	    Application.DisplayAlerts = False
	    Set Rng = Range("A2")
	    Do Until IsEmpty(Rng)
	        With Rng
	            If .MergeCells Then
	                For Each Cell In .Offset(0, 1).Resize(.MergeArea.Rows.Count, 1)
	                    TempStr = TempStr & Cell & vbCrLf
	                Next
	                .Offset(0, 1).Resize(.MergeArea.Rows.Count, 1).Merge
	                .Offset(0, 1) = Left(TempStr, Len(TempStr) - 1)
	            End If
	        End With
	        TempStr = ""
	    Set Rng = Rng.Offset(1, 0)
	    Loop
	    Application.DisplayAlerts = True
		```

	Copy the above VB code to 'Visual Basic Editor' and run. It will merge B column according to A column.
	Then change (Line #7) Set Rng = Range("A2")==>Set Rng = Range("B2") to merge C column then ===>Set Rng = Range("C2") to merge
	D column until all other columns are merged according to A column. 

4. Add two extra columns to the end of the excel form, named "CPU Results" "GPU Results" "Assignee" 

### Download and Execute API example code

Download all code Example to directory ``ExampleCodeExecutor\api_codes`` and run ``example_code_executor.py``. A
report text file will be generated in ``ExampleCodeExecutor\`` .

API Example code file name convention:

- Rule 1: If an API named *someone* has multiple example codes, the first one will be named as someone.py, second one someone_1.py, third one someone_2.py and so forth.
- Rule 2: If an API has single example code, it will be named as someone.py
- Rule 3: If an API is either a method of a class or the class body itself, it will be named as Class_name.method_name.py. If a method has multiple example codes, it follows Rule 1 as well.

Notes: Before Execute the codes and fetch the results, please make sure:

1. paddle is installed locally.
2. in ``API_to_Table/generator_saver.js`` Line 412, to customize your import/CPU/GPU environment conditions

### Assign API example code execution reports to API excel

1. Move the API Excel table to Tabel_to_Excel and named it as "all_api_table.xlsx" 
1. Move the report files (CPU report named report-cpu.txt, GPU report named report_v100_cuda9.txt) generated in the last step to Table to Excel
2. rename the Assignee table as the example in the repo and make the columns **exactly the same as** the excel in the repo
3. Finally! you can run reports-to-excel.py.  The result xlsx is named as "all_api_table_after.xlsx" 



