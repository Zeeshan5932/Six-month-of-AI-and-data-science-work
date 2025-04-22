<!-- <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Reveal Hover Effect</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: linear-gradient(135deg, #f06, #4a90e2);
            margin: 0;
            font-family: Arial, sans-serif;
        }
        .reveal-text {
            font-size: 2.5em;
            position: relative;
            color: #fff;
            cursor: pointer;
        }
        .reveal-text::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #ff6347; /* Tomato color for the reveal effect */
            transform: scaleX(1);
            transform-origin: right;
            transition: transform 0.5s ease;
        }
        .reveal-text:hover::before {
            transform: scaleX(0);
            transform-origin: left;
        }
        .reveal-text span {
            position: relative;
            color: #fff;
            z-index: 1;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="reveal-text">
        <span>Hover over me</span>
    </div>
</body>
</html>




 -->



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Glowing Tubelight Text Animation</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #000;
            margin: 0;
            font-family: 'Arial', sans-serif;
        }

        .glowing-text {
            font-size: 4em;
            color: #fff;
            position: relative;
            animation: flicker 1.5s infinite alternate;
        }

        .glowing-text::before {
            content: attr(data-text);
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            color: #fff;
            z-index: -1;
            filter: blur(10px);
            opacity: 0.7;
            animation: flicker 1.5s infinite alternate;
        }

        .glowing-text::after {
            content: attr(data-text);
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            color: #fff;
            z-index: -1;
            filter: blur(20px);
            opacity: 0.4;
            animation: flicker 1.5s infinite alternate;
        }

        @keyframes flicker {
            0% {
                opacity: 0.8;
                text-shadow: 0 0 10px #00e5ff, 0 0 20px #00e5ff, 0 0 30px #00e5ff, 0 0 40px #00e5ff, 0 0 50px #00e5ff, 0 0 60px #00e5ff, 0 0 70px #00e5ff;
            }
            50% {
                opacity: 1;
                text-shadow: 0 0 20px #00e5ff, 0 0 30px #00e5ff, 0 0 40px #00e5ff, 0 0 50px #00e5ff, 0 0 60px #00e5ff, 0 0 70px #00e5ff, 0 0 80px #00e5ff;
            }
            100% {
                opacity: 0.8;
                text-shadow: 0 0 10px #00e5ff, 0 0 20px #00e5ff, 0 0 30px #00e5ff, 0 0 40px #00e5ff, 0 0 50px #00e5ff, 0 0 60px #00e5ff, 0 0 70px #00e5ff;
            }
        }
    </style>
</head>
<body>
    <div class="glowing-text" data-text="Glowing Text">Glowing Text</div>
</body>
</html>
