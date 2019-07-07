var translationForm = document.getElementById('translationForm');

var xmlhttp = null;

function request(data, handler) {
  xmlhttp = new XMLHttpRequest();

  xmlhttp.onreadystatechange = handler;

  xmlhttp.open('POST', 'api/translate/fren', true);

  xmlhttp.setRequestHeader('Content-type', 'application/json');
  xmlhttp.send(JSON.stringify(data));
}

function translate() {
  var data = {
    'fr': document.getElementById('french').value.trim()
  };
  document.getElementById('loader').style.display = 'inline-block';
  request(data, printTranslation);
}

function translateHandler(e) {
  e.preventDefault();
  translate();
}

function keypressHandler(e) {
  if (e.which == 13) {
    e.preventDefault();
    translate();
  }
}

function printTranslation() {
  if (xmlhttp.readyState == XMLHttpRequest.DONE) {   // XMLHttpRequest.DONE == 4
    if (xmlhttp.status == 200) {
      var resp = JSON.parse(xmlhttp.responseText);
      document.getElementById('english').textContent = resp.en;
      document.getElementById('loader').style.display = 'none';
    } else {
      alert('There was an error ' + xmlhttp.status);
    }
  }
}

window.onload = function() {
  translationForm.addEventListener('submit', translateHandler);
  document.getElementById('french').addEventListener('keypress', keypressHandler);
};
