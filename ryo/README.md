

# Build

```
docker build -t kenchaaan/ryo .
```

# Run

```
docker run -d --rm -p 8080:8080 kenchaaan/ryo
```

# Use

```
curl localhost:8080

curl -X POST -H "Content-Type: application/json" \
    -d '{"x": 2}' \
    localhost:8080/api
```