function clearTextArea() {
    document.getElementById('inputText').value = '';
}

function pasteText() {
    navigator.clipboard.readText()
        .then(text => {
            document.getElementById('inputText').value += text;
        })
        .catch(err => {
            console.error('Failed to read clipboard contents: ', err);
        });
}

function uploadFile() {
    document.getElementById('documentUpload').click();
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    fetch('/convert_file_to_text/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(text => {
        document.getElementById('inputText').value += text;
    })
    .catch(error => {
        console.error('Error converting file to text:', error);
    });
}



