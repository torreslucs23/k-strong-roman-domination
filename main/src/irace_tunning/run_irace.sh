cd "$(dirname "$0")"

chmod +x target-runner.py

echo "Starting irace tuning..."
echo "Results will be saved to irace.Rdata"

irace --scenario scenario.txt

echo ""
echo "Tuning complete!"
echo "Best configurations saved in irace.Rdata"
echo ""
echo "To see results in R:"
echo "  library(irace)"
echo "  load('irace.Rdata')"
echo "  print(iraceResults)"