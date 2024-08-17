document.querySelectorAll('.drop-zone__input').forEach(inputElement => {
    const dropZoneElement = inputElement.closest(".drop-zone");

    dropZoneElement.addEventListener("click", e => {
        inputElement.click();
    });

    inputElement.addEventListener("change", e => {
        if (inputElement.files.length) {
            updateThumbnail(dropZoneElement, inputElement.files[0]);
            checkFiles();
        }
    });

    dropZoneElement.addEventListener("dragover", e => {
        e.preventDefault();
        dropZoneElement.classList.add("drop-zone--over");
    });

    ["dragleave", "dragend"].forEach(type => {
        dropZoneElement.addEventListener(type, e => {
            dropZoneElement.classList.remove("drop-zone--over");
        });
    });

    dropZoneElement.addEventListener("drop", e => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            inputElement.files = e.dataTransfer.files;
            updateThumbnail(dropZoneElement, e.dataTransfer.files[0]);
            checkFiles();
        }
        dropZoneElement.classList.remove("drop-zone--over");
    });
});

function updateThumbnail(dropZoneElement, file) {
    let thumbnailElement = dropZoneElement.querySelector(".drop-zone__prompt");
    if (dropZoneElement.querySelector(".drop-zone__thumb")) {
        thumbnailElement = dropZoneElement.querySelector(".drop-zone__thumb");
    } else {
        thumbnailElement.style.display = "none";
        const icon = document.createElement("i");
        icon.classList.add("fas", "fa-file-alt", "fa-2x");
        dropZoneElement.appendChild(icon);

        const fileName = document.createElement("p");
        fileName.textContent = file.name;
        fileName.classList.add("drop-zone__filename");
        dropZoneElement.appendChild(fileName);
    }
}

function checkFiles() {
    const fileDat = document.getElementById('fileDat').files.length;
    const fileHea = document.getElementById('fileHea').files.length;
    const submitBtn = document.getElementById('submitFilesBtn');

    if (fileDat > 0 && fileHea > 0) {
        submitBtn.disabled = false;
    } else {
        submitBtn.disabled = true;
    }
}

$(document).ready(function() {
    let selectedModel = '';

    $('#fileDat').change(function() {
        if (this.files.length > 0) {
            $('#fileHea').prop('disabled', false);
        } else {
            $('#fileHea').prop('disabled', true);
            $('#submitFilesBtn').prop('disabled', true);
            $('#analyzeBtn').prop('disabled', true);
        }
    });

    $('#fileHea').change(function() {
        if (this.files.length > 0 && $('#fileDat')[0].files.length > 0) {
            $('#submitFilesBtn').prop('disabled', false);
        } else {
            $('#submitFilesBtn').prop('disabled', true);
        }
    });

    $('#submitFilesBtn').click(function() {
        var formData = new FormData();
        formData.append('fileDat', $('#fileDat')[0].files[0]);
        formData.append('fileHea', $('#fileHea')[0].files[0]);
        formData.append('modelo', selectedModel);

        $.ajax({
            url: '/subir-archivos',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#output').html('<div class="alert alert-success">Archivos subidos correctamente</div>');
                $('#analyzeBtn').prop('disabled', false);
                plotECG([]);
            },
            error: function() {
                $('#output').html('<div class="alert alert-danger">Error al subir archivos</div>');
            }
        });
    });

    $('.model-btn').click(function() {
        $('.model-btn').removeClass('selected');
        $(this).addClass('selected');
        selectedModel = $(this).data('model');
    });

    $('#analyzeBtn').click(function() {
        let updateInterval;
        clearInterval(updateInterval);

        $('#loadingSpinner').show();

        $.ajax({
            url: '/ejecutar-deteccion',
            type: 'POST',
            data: { modelo: selectedModel },
            success: function(response) {

                $('#loadingSpinner').hide();

                if (response.result === 'Normal') {
                    $('#indicatorNormal').removeClass('btn-secondary').addClass('btn-success');
                    $('#indicatorICC').removeClass('btn-danger').addClass('btn-secondary');
                } else if (response.result === 'ICC') {
                    $('#indicatorICC').removeClass('btn-secondary').addClass('btn-danger');
                    $('#indicatorNormal').removeClass('btn-success').addClass('btn-secondary');
                }

                var trace = {
                    x: Array.from(Array(response.signal.length).keys()),
                    y: response.signal,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: 'teal' }
                };
                var layout = {
                    title: 'ECG Signal',
                    xaxis: {
                        title: 'Tiempo (muestras)',
                        range: [0, 200]
                    },
                    yaxis: { title: 'Amplitud' }
                };
                Plotly.newPlot('ecgPlot', [trace], layout);

                $('#signalInfo').html(`
                    <p><strong>Número de señales:</strong> ${response.number_of_signals}</p>
                    <p><strong>Frecuencia de muestreo de la señal:</strong> ${response.sampling_frequency} Hz</p>
                    <p><strong>Nombres de señales:</strong> ${response.signal_names.join(', ')}</p>
                    <p><strong>Longitud de la señal:</strong> ${response.signal_length} samples</p>
                `);

                let currentIndex = 0;
                updateInterval = setInterval(function () {
                    let windowSize = 200;
                    currentIndex += 10;
                    if (currentIndex + windowSize > response.signal.length) {
                        clearInterval(updateInterval);
                    } else {
                        let newX = Array.from(Array(windowSize).keys()).map(i => i + currentIndex);
                        let newY = response.signal.slice(currentIndex, currentIndex + windowSize);
                        Plotly.relayout('ecgPlot', {
                            'xaxis.range': [currentIndex, currentIndex + windowSize],
                            'data[0].x': newX,
                            'data[0].y': newY
                        });
                    }
                }, 1000);
            },
            error: function() {
                $('#loadingSpinner').hide();

                $('#output').html('<div class="alert alert-danger">Error al analizar la señal</div>');
            }
        });
    });

    plotECG([]);
});

function plotECG(signalData) {
    var trace = {
        x: Array.from(Array(signalData.length).keys()),
        y: signalData,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'teal' }
    };

    var layout = {
        title: 'Electrocardiograma',
        xaxis: {
            title: 'Tiempo (muestras)',
            range: [0, 200]
        },
        yaxis: { title: 'Amplitud' }
    };

    Plotly.newPlot('ecgPlot', [trace], layout);
}
