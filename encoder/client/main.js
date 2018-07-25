
var attrs = ['educated at', 'sex or gender', 'country of citizenship', 'native language', 'position held', 'award received', "religion", 'participant of', 'member of political party', 'work location', 'place of death', 'place of birth', 'cause of death', 'lifespan', 'date of birth'];
$( document ).ready(function(){

	var retrieveUrl = "/attrvalues";

	$.get(retrieveUrl, function(attrvalues, status){
		
		var select = "<div class=\"row\">";

		for (var i=0; i<attrs.length; i++) {
			select+="<select id=\"a" + i.toString() + "\" name=\"" + attrs[i] + "\"><option value=\"-1\">-" + attrs[i] + "-</option>";
			var obj=attrvalues[i];
			//var keysSorted = Object.keys(obj).sort(function(a,b){return obj[a]-obj[b]})
			jQuery.each(obj, function(num, lbl) {
				if (lbl) select += "<option value=\"" + num + "\">" + lbl + "</option>";
			});
			select+="</select>";
		}
		$("#choice").html(select);
		for (var i=0; i<attrs.length; i++) {
			var select_id = "#a" + i.toString();
			$(select_id).html($(select_id + " option").sort(function (a, b) {
				return a.text == b.text ? 0 : a.text < b.text ? -1 : 1
			}));
			$(select_id).val('-1');
		}
	});

});

var generateStereotypes = function(params) {
        var reqUrl = '/demo';
        $(".info").text("Generating your stereotypes. This might take a while (the expected waiting time is under a minute).");

        $.get(reqUrl, params, function(response, status){
                $(".info").text("");
                $(".info").append(response.replace(/\n/g, '<br/><br/>'));
        });

}

var randomExample = function(){
	generateStereotypes({});
}

var userInputs = function(){
        var params = {};
	for (var i=0; i<attrs.length; i++){
		var id="#a" + i.toString();
		if ($(id).val()!='-1')
		{
			params[i.toString()]=$(id).val();
		}
	}
	generateStereotypes(params);
}
