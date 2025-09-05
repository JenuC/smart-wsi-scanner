$client = New-Object System.Net.Sockets.TcpClient("127.0.0.1", 5000)
$stream = $client.GetStream()
$writer = New-Object System.IO.StreamWriter($stream)
$writer.Write("getxy___")
$writer.Flush()

$reader = New-Object System.IO.StreamReader($stream)
$response = $reader.ReadToEnd()
$response

$reader.Close()
$writer.Close()
$stream.Close()
$client.Close()