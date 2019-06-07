var translate = document.getElementById('translate');

var xmlhttp = null;

function request(data, handler) {
  xmlhttp = new XMLHttpRequest();

  xmlhttp.onreadystatechange = handler;

  xmlhttp.open('POST', 'api/translate/fren', true);

  xmlhttp.setRequestHeader('Content-type', 'application/json');
  xmlhttp.send(JSON.stringify(data));
}

function translate_handler(e) {
  e.preventDefault();
  var data = {
    'fr': document.getElementById('french').value
  };
  request(data, print_translation);
}

function print_translation() {
  if (xmlhttp.readyState == XMLHttpRequest.DONE) {   // XMLHttpRequest.DONE == 4
   if (xmlhttp.status == 200) {
     var resp = JSON.parse(xmlhttp.responseText);
     document.getElementById('english').textContent = resp.en;
   } else {
     alert('There was an error ' + xmlhttp.status);
   }
  }
}

window.onload = function() {
    translationForm.addEventListener('submit', translate_handler);
};
