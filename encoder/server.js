var exec = require('child_process').exec;
var express = require('express');
var app = express();
var fs = require('fs');
var randomstring = require("randomstring");


//var attrs=['education', 'gender', 'citizenship', 'language', 'work', 'award', "religion", 'participantof', 'politicalparty', 'sportsteam'];

var attrs=['educated at', 'sex or gender', 'country of citizenship', 'native language', 'position held', 'award received', "religion", 'participant of', 'member of political party', 'work location', 'place of death', 'place of birth', 'cause of death', 'lifespan', 'date of birth'];

app.get('/', function(req, res){
    res.sendFile('index.html', {root:'./client'});
});

app.use('/', express.static('client/'));

app.get('/attrvalues', function(req, res){
    fs.readFile('src/labels.json', 'utf8', function(err, data){
         if (err) throw err;
         var obj = JSON.parse(data);
         res.send(obj);
    });
});

app.get('/demo', function(req, res){
    var inputs = [];
    for (var i=0; i<attrs.length; i++){
	if (req.param(i.toString())){
            var value=req.param(i.toString());
            if (value[0]=='Q')
                inputs.push('http://www.wikidata.org/entity/' + value); // + '\t';
            else
                inputs.push(req.param(i.toString())); // + '\t';
        } else
            inputs.push(' '); // += ' \t';
    }
    var input = inputs.join('\t') + '\n';
    if (input.replace(/\s/g, '').length){
        console.log('user inputs');
        var code = randomstring.generate();
        fs.writeFile('demo_data/politician/test' + code + '.txt', input, function(err){});
        var shell_command = './run_single_web_example.sh ' + code;
    } else {
        var shell_command = './run_single_random_example.sh';
        console.log('random sample');
    }
    // then run the single example script in shell
    exec(shell_command, function (error, stdout, stderr) {
        console.log('done executing');
        if (error !== null) {
            console.log('exec error: ' + error);
            res.send('exec error: ' + error);
        } else {
            console.log(stdout);
            res.send(stdout);
        }
    });

    //res.send();
});

app.listen(8484, function() {
	console.log('started stereotypes server backend');
});
