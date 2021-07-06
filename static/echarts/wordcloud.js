var d3 = window.d3;
d3.text("/echarts/article-abstracts.txt").then((words) => {
    var stopwords = new Set(
        "i,me,my,myself,we,us,our,ours,ourselves,you,your,yours,yourself,yourselves,he,him,his,himself,she,her,hers,herself,it,its,itself,they,them,their,theirs,themselves,what,which,who,whom,whose,this,that,these,those,am,is,are,was,were,be,been,being,have,has,had,having,do,does,did,doing,will,would,should,can,could,ought,i'm,you're,he's,she's,it's,we're,they're,i've,you've,we've,they've,i'd,you'd,he'd,she'd,we'd,they'd,i'll,you'll,he'll,she'll,we'll,they'll,isn't,aren't,wasn't,weren't,hasn't,haven't,hadn't,doesn't,don't,didn't,won't,wouldn't,shan't,shouldn't,can't,cannot,couldn't,mustn't,let's,that's,who's,what's,here's,there's,when's,where's,why's,how's,a,an,the,and,but,if,or,because,as,until,while,of,at,by,for,with,about,against,between,into,through,during,before,after,above,below,to,from,up,upon,down,in,out,on,off,over,under,again,further,then,once,here,there,when,where,why,how,all,any,both,each,few,more,most,other,some,such,no,nor,not,only,own,same,so,than,too,very,say,says,said,shall".split(
            ","
        )
    );
    words = words
        .trim()
        .split(/[\s.]+/g)
        .map((w) => w.replace(/^[“‘"\-—()[\]{}]+/g, ""))
        .map((w) => w.replace(/[;:.!?()[\]{},"'’”\-—]+$/g, ""))
        .map((w) => w.replace(/['’]s$/g, ""))
        .map((w) => w.substring(0, 30))
        .map((w) => w.toLowerCase())
        .filter((w) => w && !stopwords.has(w));

    var data = d3
        .rollups(
            words,
            (group) => group.length,
            (w) => w
        )
        .sort(([, a], [, b]) => d3.descending(a, b))
        .slice(0, 250)
        .map(([name, value]) => ({ name, value }));

    // https://github.com/ecomfe/echarts-wordcloud/blob/master/example/wordCloud.html
    var chart = echarts.init(document.getElementById('word-cloud-div'));


    var option = {
        tooltip: {},
        series: [ {
            type: 'wordCloud',
            gridSize: 2,
            sizeRange: [12, 100],
            rotationRange: [-35, 55],
            shape: 'pentagon',
            width: 600,
            height: 400,
            drawOutOfBound: false,
            textStyle: {
                color: function () {
                    return 'rgb(' + [
                        Math.round(Math.random() * 160),
                        Math.round(Math.random() * 160),
                        Math.round(Math.random() * 160)
                    ].join(',') + ')';
                }
            },
            emphasis: {
                textStyle: {
                    shadowBlur: 10,
                    shadowColor: '#333'
                }
            },
            data: data.sort(function (a, b) {
                return b.value  - a.value;
            })
        }]
    };
    chart.setOption(option);

    window.onresize = chart.resize;

});


