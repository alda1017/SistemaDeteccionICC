const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const mysql = require("mysql2");
const multer = require("multer");
const path = require("path");
const { execFile } = require('child_process');
const session = require('express-session');


const app = express();

app.use(session({
    secret: 'tu_secreto_secreto',
    resave: false,
    rolling: true,
    saveUninitialized: true,
    cookie: {
        secure: false,
        maxAge: 1000 * 60 * 15
    }
}));

app.get('/login.html', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'public', 'login.html'));
});

app.get('/formulario.html', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'public', 'formulario.html'));
});

function checkAuthentication(req, res, next) {
    if (req.session.authenticated) {
        next();
    } else {
        res.redirect('/login.html');
    }
}

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const uploadPath = path.join(__dirname, 'uploads');
        cb(null, uploadPath);
    },
    filename: function (req, file, cb) {
      cb(null, file.originalname);
    }
  });
const upload = multer({ storage: storage });

const pool = mysql.createPool({
    host: 'localhost',
    user: 'alda7',
    password: 'fireStone!02resi2',
    database: 'telemed242',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});

pool.getConnection((err, connection) => {
    if (err) {
        console.error("No se pudo conectar a la base de datos:", err);
        return;
    }
    console.log("Conectado exitosamente a la base de datos");
    connection.release();
});

app.use('/privado', checkAuthentication, express.static('public'));
app.use(express.static(path.join(__dirname, 'public/login.html')));
app.use('/privado/uploads', checkAuthentication, express.static(path.join(__dirname, 'uploads')));
app.use(cors());
app.use(bodyParser.json());
app.use(express.urlencoded({ extended: true }));


app.post('/enviar-datos', (req, res) => {
    const { nombre, correo, password } = req.body;

    pool.query('INSERT INTO usuarios (nombre, correo, password) VALUES (?, ?, ?)', [nombre, correo, password], (err, results) => {
        if (err) {
            console.error('Error al insertar en la base de datos:', err);
            res.status(500).send('Error al insertar los datos');
        } else {
            console.log('Datos insertados con éxito:', results);
            res.status(200).send('Datos recibidos con éxito');
        }
    });
});

app.post('/subir-archivo', upload.single('archivo'), (req, res) => {
    const nombreArchivo = req.file.filename;
    const rutaArchivo = req.file.path;
    const tipoArchivo = req.file.mimetype;
   
    const sql = 'INSERT INTO archivos (nombre_archivo, ruta_archivo, tipo_archivo) VALUES (?, ?, ?)';
    pool.query(sql, [nombreArchivo, rutaArchivo, tipoArchivo], (err, results) => {
      if (err) {
        console.error('Error al insertar en la base de datos:', err);
        res.status(500).send('Error al insertar los datos');
      } else {
        console.log('Datos del archivo insertados con éxito:', results);
        res.status(200).send('Archivo subido y datos guardados con éxito');
      }
    });
});

app.post('/subir-archivos', upload.fields([
    { name: 'fileDat', maxCount: 1 },
    { name: 'fileHea', maxCount: 1 }
]), (req, res) => {
    if (!req.files.fileDat || !req.files.fileHea) {
        return res.status(400).send('Archivos no proporcionados.');
    }

    const modeloSeleccionado = req.body.modelo;
    req.session.modelo = modeloSeleccionado;
    const fileDat = req.files.fileDat[0];
    const baseName = fileDat.filename.replace('.dat', '');

    req.session.baseName = baseName;

    const fileHea = req.files.fileHea[0];
    const nombreArchivoDat = fileDat.filename;
    const rutaArchivoDat = fileDat.path;
    const tipoArchivoDat = fileDat.mimetype;
    const nombreArchivoHea = fileHea.filename;
    const rutaArchivoHea = fileHea.path;
    const tipoArchivoHea = fileHea.mimetype;

    const sqlDat = 'INSERT INTO archivos (nombre_archivo, ruta_archivo, tipo_archivo) VALUES (?, ?, ?)';
    pool.query(sqlDat, [nombreArchivoDat, rutaArchivoDat, tipoArchivoDat], (err, results) => {
        if (err) {
            console.error('Error al insertar el archivo .dat en la base de datos:', err);
            return res.status(500).send('Error al insertar los datos del archivo .dat');
        }

        const sqlHea = 'INSERT INTO archivos (nombre_archivo, ruta_archivo, tipo_archivo) VALUES (?, ?, ?)';
        pool.query(sqlHea, [nombreArchivoHea, rutaArchivoHea, tipoArchivoHea], (err, results) => {
            if (err) {
                console.error('Error al insertar el archivo .hea en la base de datos:', err);
                return res.status(500).send('Error al insertar los datos del archivo .hea');
            }
            res.send('Archivos subidos y datos guardados con éxito.');
        });
    });
});

app.post('/iniciar-sesion', (req, res) => {
    const { correo, password } = req.body;
    pool.query('SELECT * FROM usuarios WHERE correo = ?', [correo], (err, results) => {
        if (err) {
            console.error('Error al consultar la base de datos:', err);
            res.status(500).send('Error al procesar la solicitud');
            return;
        }
        if (results.length > 0) {
            if (results[0].password === password) {
                req.session.authenticated = true;
                res.send('OK');
            } else {
                res.status(401).send('Credenciales incorrectas');
            }
        } else {
            res.status(404).send('Usuario no encontrado');
        }
    });
});

app.get('/', (req, res) => {
    if (req.session.authenticated) {
        res.redirect('/privado/index.html');
    } else {
        res.redirect('/login.html');
    }
});

app.get('/consultar', (req, res) => {
    pool.query('SELECT * FROM usuarios', (err, results) => {
        if (err) {
            console.error('Error al consultar la base de datos:', err);
            res.status(500).send('Error al consultar los datos');
        } else {
            console.log('Usuarios consultados con éxito');
            res.status(200).json(results);
        }
    });
});

app.get('/privado/archivos', checkAuthentication, (req, res) => {
    pool.query('SELECT * FROM archivos', (err, results) => {
      if (err) {
        console.error('Error al consultar la base de datos:', err);
        res.status(500).send('Error al consultar los datos');
      } else {
        console.log('Archivos consultados con éxito');
        res.status(200).json(results);
      }
    });
});

app.post('/ejecutar-deteccion', (req, res) => {
    if (!req.session.baseName || !req.body.modelo) {
        return res.status(400).send('No se ha proporcionado el nombre del archivo o el modelo.');
    }

    console.log("Endpoint /ejecutar-deteccion fue llamado");
    console.log("Modelo seleccionado para la clasificación:", req.body.modelo);

    let scriptPath;
    switch (req.body.modelo) {
        case "SVM":
            scriptPath = path.join(__dirname, 'Python', 'SVM_sistema_deteccion_ICC.py');
            break;
        case "KNN":
            scriptPath = path.join(__dirname, 'Python', 'KNN_sistema_deteccion_ICC.py');
            break;
        case "RN":
            scriptPath = path.join(__dirname, 'Python', 'RN_sistema_deteccion_ICC.py');
            break;
        default:
            return res.status(400).send('Modelo seleccionado no válido.');
    }

    const fullPath = path.join('./server/uploads/', req.session.baseName);

    execFile('python3.11', [scriptPath, fullPath], { maxBuffer: 10 * 1024 * 1024 }, (error, stdout, stderr) => {
        console.log("Listo");

        if (error) {
            console.error('Error al ejecutar el script de Python:', error);
            return res.status(500).json({ error: error.message });
        }
        if (stderr.includes("ERROR") || stderr.includes("Exception") || stderr.includes("Traceback")) {
            console.error('Error crítico en el script de Python:', stderr);
            return res.status(500).json({ error: stderr });
        }

        const cleanOutput = stdout.split('\n').filter(line => {
            return !/\d+\/\d+/.test(line);
        }).join('\n');

        try {
            const result = JSON.parse(cleanOutput);
            res.json(result);
        } catch (parseError) {
            console.error('Error parsing Python output:', parseError);
            res.status(500).send('Error parsing Python output');
        }
    });
});

app.listen(5001, function(){
   console.log("Servidor corriendo en el puerto 5001");
});
