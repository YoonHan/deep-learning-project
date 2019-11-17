chrome.webRequest.onBeforeRequest.addListener(function(request) {
	console.log(request['url']);
    //console.log("version 1.1 000000000000");
	var level = request['url'].substring(request['url'].indexOf("Level"));
	if (level == 'Level1-1.json' ||
		level == 'Level1-2.json' ||
		level == 'Level1-3.json' ||
		level == 'Level1-4.json' ||
		level == 'Level1-5.json' ||
		level == 'Level1-6.json' ||
		level == 'Level1-7.json' ||
		level == 'Level1-8.json' ||
		level == 'Level1-9.json' ||
		level == 'Level1-10.json' ||
           level == 'Level1-11.json' ||
		level == 'Level1-12.json' ||
		level == 'Level1-13.json' ||
		level == 'Level1-14.json' ||
		level == 'Level1-15.json' ||
		level == 'Level1-16.json' ||
		level == 'Level1-17.json' ||
		level == 'Level1-18.json' ||
		level == 'Level1-19.json' ||
		level == 'Level1-20.json' ||
           level == 'Level1-21.json' ||
		   level == 'Level2-1.json' ||
		level == 'Level2-2.json' ||
		level == 'Level2-3.json' ||
		level == 'Level2-4.json' ||
		level == 'Level2-5.json' ||
		level == 'Level2-6.json' ||
		level == 'Level2-7.json' ||
		level == 'Level2-8.json' ||
		level == 'Level2-9.json' ||
		level == 'Level2-10.json' ||
           level == 'Level2-11.json' ||
		level == 'Level2-12.json' ||
		level == 'Level2-13.json' ||
		level == 'Level2-14.json' ||
		level == 'Level2-15.json' ||
		level == 'Level2-16.json' ||
		level == 'Level2-17.json' ||
		level == 'Level2-18.json' ||
		level == 'Level2-19.json' ||
		level == 'Level2-20.json' ||
           level == 'Level2-21.json' ||
		   level == 'Level3-1.json' ||
		level == 'Level3-2.json' ||
		level == 'Level3-3.json' ||
		level == 'Level3-4.json' ||
		level == 'Level3-5.json' ||
		level == 'Level3-6.json' ||
		level == 'Level3-7.json' ||
		level == 'Level3-8.json' ||
		level == 'Level3-9.json' ||
		level == 'Level3-10.json' ||
           level == 'Level3-11.json' ||
		level == 'Level3-12.json' ||
		level == 'Level3-13.json' ||
		level == 'Level3-14.json' ||
		level == 'Level3-15.json' ||
		level == 'Level3-16.json' ||
		level == 'Level3-17.json' ||
		level == 'Level3-18.json' ||
		level == 'Level3-19.json' ||
		level == 'Level3-20.json' ||
           level == 'Level3-21.json' 
		) {
		return { redirectUrl: chrome.extension.getURL('levels/' + level) };
	}
}, {
	urls: [
		"http://*.appspot.com/*"
	],
	types: ["main_frame", "sub_frame", "stylesheet", "script", "image", "object", "xmlhttprequest", "other"]
}, ['blocking']);
