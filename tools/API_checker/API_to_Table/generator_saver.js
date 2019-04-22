

function ApiObject(){


    this.api_name="";
    this.func_api_path="";
    this.api_type="";
    this.func_api_arguments_str="";
    this.explain_contents="";
    this.param_ul_str="";
    this.return_str="";
    this.return_type_str="";
    this.raise_str="";
    this.example_code="";
    this.example_code_to_save=[];



    this.save_example_code=save_example_code;

    function save_example_code(){
                                        
        //alert("save the example code in "+this.api_name+"?");
       


        for(var x=0;x<this.example_code_to_save.length;x++){

            var blob = new Blob([this.example_code_to_save[x]], {type: "text/plain;charset=utf-8"});
            if(x>0){
              saveAs(blob, this.api_name+"_"+x+".py");
            }else{
               saveAs(blob, this.api_name+".py");
            }
        }


    }
}
/*
a brief copy of APIObject , for downloading codes
*/

function ApiObject_concise(){


    this.api_name="";
    /*
    this.func_api_path="";
    this.api_type="";
    this.func_api_arguments_str="";
    this.explain_contents="";
    this.param_ul_str="";
    this.return_str="";
    this.return_type_str="";
    this.raise_str="";
    this.example_code="";
    */
    this.example_code_to_save=[];



    this.save_example_code=save_example_code;

    function save_example_code(){
                                        
        //alert("concise-save the example code in "+this.api_name+"?");
        

        for(var x=0;x<this.example_code_to_save.length;x++){

            //alert(this.example_code_to_save[x]);

            var blob = new Blob([this.example_code_to_save[x]], {type: "text/plain;charset=utf-8"});
            
            if(x>0){
                
                saveAs(blob, this.api_name+"_"+x+".py");

            }else{
                
                saveAs(blob, this.api_name+".py");
            }
        }


    }
}


function api_row_maker(api_object){

    //alert(JSON.stringify(api_object));
    
    var new_tr_ele=document.createElement("tr");
    if(api_object.api_type=="class"){
        new_tr_ele.style.backgroundColor="#FFCC33";
    }
    if(api_object.api_type=="method"){
        new_tr_ele.style.backgroundColor="#CCFF33";
    }
    if(api_object.api_type=="attribute"){
        new_tr_ele.style.backgroundColor="#FFFF99";
    }


    //td
    for (x in api_object){
        if(x != "save_example_code" && x!="example_code_to_save" && x!="properties_and_methods"){
            var new_td_ele=document.createElement("td");
            new_td_ele.className=x;
            new_td_ele.innerHTML=api_object[x];
            new_tr_ele.appendChild(new_td_ele);
        }
    }
    

    var new_td_ele=document.createElement("td");

    if(api_object.example_code_to_save.length>0){
        var saveBtn=document.createElement("input");
        saveBtn.className="save-button";
        saveBtn.value="Save the example Code";
        saveBtn.type="button";
        saveBtn.addEventListener("click", function(){api_object.save_example_code()}); 

        new_td_ele.appendChild(saveBtn);
    }else{
        new_td_ele.innerHTML="没给示例代码呢，不能下载╮(╯_╰)╭"
    }

    new_tr_ele.appendChild(new_td_ele);
    


    var table_ele=document.getElementById("api_table");
    table_ele.appendChild(new_tr_ele);
    


}

function api_file_select_options(){

    var api_file_select=document.getElementById("api_file_select");

    for (var i=0;i<files.length;i++){
        var option=document.createElement("option");
        option.text=files[i];
        option.value=files[i];
        api_file_select.add(option);
    }
}


function func_method_api_handler(api,analysed_api,is_method,is_class_body){
    /**
     * Analyse standard function / methods API,
     * fill the api object with extracted contents then
     * return it back to dispaly
     * */ 

    //3. api arguments
    if(is_class_body==true&&is_method==false){
        var func_api_dt=api.querySelector("dt");
        //alert(func_api_dt.innerHTML);
    }
    else if(is_method==true&&is_class_body==false){
        var func_api_dt=api.querySelector("dl[class=method] dt,dl[class=attribute] dt");
        
    }
    else
    {
        var func_api_dt=api.querySelector("dl[class=function] dt");
    }
    
    var func_api_arguments = func_api_dt.getElementsByTagName("em");
    var func_api_arguments_str="";
    for(var k=0;k<func_api_arguments.length;k++){
        if(is_class_body==true&&k==0){
            continue;
        }
        func_api_arguments_str+=func_api_arguments[k].innerHTML;
        func_api_arguments_str+=", ";
    }

    analysed_api.func_api_arguments_str=func_api_arguments_str;

    //alert(func_api_arguments_str);//ok
    if(is_class_body==true&&is_method==false){
        analysed_api.api_type="class";
        //alert(func_api_dt.innerHTML);
    }
    else if(is_method==true&&is_class_body==false){
        analysed_api.api_type="method";
        var attr_test=api.querySelector("dl[class=attribute] dt");
        if(attr_test){
            analysed_api.api_type="attribute";
        }
        
    }
    else
    {
        analysed_api.api_type="function";
    }
    

    



    //4. api explanatory contents
    
    if(is_class_body==true&&is_method==false){
        var func_api_dd=api.querySelector("dd");
        //alert(func_api_dd.innerHTML);
    }
    else if(is_method==true&&is_class_body==false){
        var func_api_dd=api.querySelector("dl[class=method] dd,dl[class=attribute] dd");
    }
    else
    {
        var func_api_dd=api.querySelector("dl[class=function] dd");
    }
    //var explain_end=0;
    //alert(func_api_dd.childNodes.length);

    var explain_contents="";

    for(var k=0;k<func_api_dd.childNodes.length;k++){
        //alert(k);
        //alert(func_api_dd.childNodes[k].tagName);
        //alert(func_api_dd.childNodes[k].className);

        if(func_api_dd.childNodes[k].tagName=="TABLE"&&func_api_dd.childNodes[k].className=="docutils field-list"){
            explain_end=k;
            
            break;
        }else{
            var this_paragraph=func_api_dd.childNodes[k].innerHTML;
            if(this_paragraph){
                explain_contents+=this_paragraph;
            }
        }
    }
    // childNodes[0-k-1]:explanatory
    // childNodes[k-length-1]:Parameters,returns, raises,examples

    //alert(explain_contents);//ok
    analysed_api.explain_contents=explain_contents;



    
    //5. parameters list
    //tbody --> tr parameters
    //      --> tr Returns
    //      --> tr Raises
    var func_api_dd_table=api.querySelector("dl[class=function] dd table");
    if(is_class_body==true&&is_method==false){
        var func_api_dd=api.querySelector("dd table");
        //alert(func_api_dd.innerHTML);
    }
    else if(is_method==true&&is_class_body==false){
        var func_api_dd=api.querySelector("dl[class=method] dd table");
    }
    else
    {
        var func_api_dd=api.querySelector("dl[class=function] dd table");
    }



    if(func_api_dd_table!=null){
        var func_api_dd_tbody=func_api_dd_table.getElementsByTagName("TBODY")[0];
        var trs=func_api_dd_tbody.getElementsByTagName("TR");
    }else{ //no argument table
        var func_api_dd_tbody="";
        var trs=[]
    }

    analysed_api.param_ul_str="Not Given";
    analysed_api.return_str="Not Given";
    analysed_api.return_type_str="Not Given";
    analysed_api.raise_str="Not Given";
    analysed_api.example_code="Not Given";
    analysed_api.example_code_to_save="";





    for(var k=0;k<trs.length;k++){

        if(trs[k].querySelector("th[class=field-name]").innerHTML=="Parameters:"){
            
            var param_ul=trs[k].querySelector("td[class=field-body] ul");
            //alert(param_ul.innerHTML);//ok
            var param_ul_str="";
            if(param_ul!=null){
                param_ul_str=param_ul.innerHTML;
                analysed_api.param_ul_str=param_ul_str;
            }

            continue;

        }


        if(trs[k].querySelector("th[class=field-name]").innerHTML=="Returns:"){
            
            var return_ele=trs[k].querySelector("td[class=field-body]");
            
            
            var return_str="";
            if(return_ele!=null){
                return_str=return_ele.innerHTML;
                analysed_api.return_str=return_str;
            }

            continue;

        }


        if(trs[k].querySelector("th[class=field-name]").innerHTML=="Return type:"){
            
            var return_type_ele=trs[k].querySelector("td[class=field-body]");
            
            
            var return_type_str="";
            if(return_type_ele!=null){
                return_type_str=return_type_ele.innerHTML;
                analysed_api.return_type_str=return_type_str;
            }

            continue;

        }


        if(trs[k].querySelector("th[class=field-name]").innerHTML=="Raises:"){
            
            var raise_ele=trs[k].querySelector("td[class=field-body]");
            
            
            var raise_str="";
            if(raise_ele!=null){
                raise_str=raise_ele.innerHTML;
                analysed_api.raise_str=raise_str;
            }
            continue;

        }



    }

    //6. example code
    var func_api_dd_code_python=api.getElementsByClassName("highlight-python");
    var func_api_dd_code_default=api.getElementsByClassName("highlight-default");


    var example_code_to_save=[];

    for(var k=0;k<func_api_dd_code_python.length;k++){
        
        if(analysed_api.example_code=="Not Given"){
            analysed_api.example_code="";
        }

        var example_code="# <em>Example code No.<em> "+k+"\n"+func_api_dd_code_python[k].innerHTML;

        analysed_api.example_code+=example_code;

        
        example_code_to_save.push(func_api_dd_code_python[k].innerText);

    }


    for(var k=0;k<func_api_dd_code_default.length;k++){
        
        if(analysed_api.example_code=="Not Given"){
            analysed_api.example_code="";
        }

        var example_code="# <em>Example code No.<em> "+k+"\n"+func_api_dd_code_default[k].innerHTML;

        analysed_api.example_code+=example_code;

        example_code_to_save.push(func_api_dd_code_default[k].innerText);
       
        
        

    }

    for(var x=0;x<example_code_to_save.length;x++){

        var lines=example_code_to_save[x].split('\n');
        for (line in lines){
            if(lines[line].indexOf(">>>")==0){
                lines[line]=lines[line].slice(4);
            }
        }
        example_code_to_save[x]=lines.join("\n");

        if(example_code_to_save[x].indexOf("import paddle.fluid as fluid")==-1){
            example_code_to_save[x]="\nimport paddle.fluid as fluid\n\nimport os\n\nos.environ[\"CUDA_VISIBLE_DEVICES\"] = \"0\" \n"+example_code_to_save[x];

        }
        example_code_to_save[x]+="\n\nprint('successful')";
    }

    analysed_api.example_code_to_save=example_code_to_save;


    return analysed_api;


}




var files=['io_en.html', //0
'fluid_en.html', //1
'profiler_en.html', //2
'backward_en.html', //3
'data_feeder_en.html', //4
'metrics_en.html', //5
'clip_en.html', //6
'regularizer_en.html',//7 
'nets_en.html', //8
'executor_en.html',//9 
'initializer_en.html',//10 
'transpiler_en.html', //11
'average_en.html', //12
'layers_en.html', //13
'optimizer_en.html']//14


//var filename=files[13];

var api_objs_per_file=[];



$(document).ready(function(){


            api_file_select_options();
            //for(var file_i=0;file_i<files.length;file_i++){

                //var filename=files[file_i];
                //alert(filename);
            var form_btn=document.getElementById("submit-button");
            form_btn.onclick=function(){

                //form_btn.disabled="true";

                var filename=document.getElementById("api_file_select").value;



                $("#contents").append("<div width='100%' class='file_hinter'>"+filename+"</div>")
                $("#contents").append("<div id='api_file_div'></div>")

                //alert("#"+i+"_div")
                var api_counter=0;
                
                


                $("#api_file_div").load("api_en_html/"+filename+" #doc-content div .section",
                    function(){

                                //alert(filename+" loading....");

                                $(".btn-group").remove();
                                $(".headerlink").remove();
                                $(".viewcode-link").remove();
                                var apis=this.getElementsByClassName("section");
                                //alert(apis.length);

                                apis[0].remove();//remove first title
                                //this.innerHTML=apis[0].innerHTML;
                                



                                for(var j=0;j<apis.length;j++){
                                    
                                    var api=apis[j];
                                    

                                    var analysed_api=new ApiObject();

                                    // for function type
                                    //1. api name
                                    var api_name_ele=null;
                                    if(filename=="layers_en.html"){

                                        //  var layers_subcategories=['control-flow', 'device', 'io', 'nn', 'ops', 'tensor', 'learning-rate-scheduler', 'detection', 'metric-op'];

                                        /*

                                          -- control flow
                                                -- api-wrapper
                                                    --h2
                                           but, normal api

                                          -- control flow
                                                -- api-wrapper
                                                -- array-length : api[?] .api-wrapper h3 can be found
                                                    -- api-wrapper
                                                        -- h3
                                        */          
                                        api_name_ele=api.querySelector(".api-wrapper h3");




                                    }else{

                                        api_name_ele=api.querySelector(".api-wrapper h2");

                                        
                                    }

                                    if(!api_name_ele){

                                            //alert(api.innerHTML);
                                            continue;
                                            
                                    }
                                        
                                    var api_name=api_name_ele.innerHTML;
                                    


                                    //alert(api_name); //ok
                                    analysed_api.api_name=api_name;

                                    //2. api path

                                    //2.1 function type
                                    var func_api_path_ele=api.querySelector("dl[class=function] dt .descclassname");
                                    //2.2 class type
                                    var class_api_path_ele=api.querySelector("dl[class=class] dt .descclassname");
                                    
                                    
                                    if(func_api_path_ele!=null){


                                        //func type processing

                                        func_api_path=func_api_path_ele.innerHTML;
                                        analysed_api.func_api_path=func_api_path;
                                        
                                        analysed_api=func_method_api_handler(api,analysed_api,false,false);


                                    }else if(class_api_path_ele!=null){ // is a class type
                                        
                                        class_api_path=class_api_path_ele.innerHTML;
                                        analysed_api.func_api_path=class_api_path;
                                        //alert(class_api_path);

                                        /**
                                         * 
                                         * 1. class body analysis
                                         */
                                        
                                        // get the class api part with the methods/properties hollowed out
                                        var class_api_dd=api.querySelector("dl[class=class] dd");
                                        var class_api_dt=api.querySelector("dl[class=class] dt");
                                        var class_only_body=document.createElement("div");
                                        class_only_body.appendChild(class_api_dt);
                                        var class_only_body_dd=document.createElement("dd");
                                        class_only_body.appendChild(class_only_body_dd);
                                        for(var k=0;k<class_api_dd.childNodes.length;k++){
                            
                                            // when the reach a new property/method element in a class api
                                            if(class_api_dd.childNodes[k].tagName=="DL"&&
                                            (class_api_dd.childNodes[k].className=="method"||class_api_dd.childNodes[k].className=="attribute")){
                                                
                                                break;
                                            }
                                            else{
                                                class_only_body_dd.appendChild(class_api_dd.childNodes[k]);
                                            }

                                        }
                                        //alert(class_only_body.innerHTML);
                                        analysed_api=func_method_api_handler(class_only_body,analysed_api,false,true);

                                        


                                        
                                    
                                    }else{
                                        //no identified type
                                        continue;
                                    }
                                    

                                    

                                    //alert(analysed_api.func_api_path); //ok
                                    

                                    
                                    
                                    
                                    
                                        
                                    



                                    //alert(JSON.stringify(analysed_api));
                                    api_row_maker(analysed_api);
                                    var temp_api_obj= new ApiObject_concise();
                                    temp_api_obj.api_name=analysed_api.api_name;
                                    temp_api_obj.example_code_to_save=analysed_api.example_code_to_save;
                                    api_objs_per_file.push(temp_api_obj);
                                    api_counter+=1;



                                    if(analysed_api.api_type=="function"){
                                        continue;
                                    }else{
                                        /**
                                         * 
                                         * class method/attribute analysis
                                         * 
                                         */
                                        analysed_api.properties_and_methods=api.querySelectorAll("dl[class=method],dl[class=attribute]");
                                        
                                        var prop_methods_per_class=[];
                                        for(var u=0;u<analysed_api.properties_and_methods.length;u++){
                                            var prop_or_method=analysed_api.properties_and_methods[u];
                                            //alert(prop_or_method.innerHTML)
                                            var prop_or_method_name=prop_or_method.querySelector("dl[class=method] dt .descname,dl[class=attribute] dt .descname");

                                            var analysed_prop_or_method=new ApiObject();
                                            analysed_prop_or_method.api_name=analysed_api.api_name+". "+prop_or_method_name.innerHTML;

                                            analysed_prop_or_method=func_method_api_handler(prop_or_method,analysed_prop_or_method,true,false);


                                            
                                            api_row_maker(analysed_prop_or_method);
                                            var temp_api_obj= new ApiObject_concise();
                                            temp_api_obj.api_name=analysed_prop_or_method.api_name;
                                            temp_api_obj.example_code_to_save=analysed_prop_or_method.example_code_to_save;
                                            api_objs_per_file.push(temp_api_obj);
                                            prop_methods_per_class.push(temp_api_obj);
                                            

                                            
                                        }
                                    }




                                
                            }//single file api loop


                                 
                            document.getElementById("report-div-p").innerHTML+=filename+" is analysed ! Number of Analysed API: "+api_counter+"<br>";
                            
                            var down_btn=document.getElementById("download-button");
                            down_btn.disabled="";
                            down_btn.onclick=function(){
                                        //alert(api_objs_per_file.length);
                                        for(var t=0;t<api_objs_per_file.length;t++){

                                            //alert(api_objs_per_file[t].example_code_to_save);
                                            
                                            var this_api=api_objs_per_file[t];
                                            //alert(this_api.api_name);
                                            //alert("downloading..."+this_api.api_name);
                                            this_api.save_example_code();
                                        }
                            }



                    });//load
                }// on submit
            //}

            
        

        
        

    

    




});
