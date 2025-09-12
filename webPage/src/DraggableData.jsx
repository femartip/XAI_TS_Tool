import React from "react";
import { Line } from "react-chartjs-2";
import "chartjs-plugin-dragdata";




const DraggableGraph = ({ dataSetCurrent, setDataCurrent, dataSetSimp, dataSetOriginal, updateData, lineColorCurr, lineColorSimp, lineColorOrg }) => {
  if (!dataSetCurrent) {
    dataSetCurrent = [];
  }
  if (!dataSetSimp) {
    dataSetSimp = [];
  }
  if (!dataSetOriginal) {
    dataSetOriginal = [];

  }
  let allData = [];
  if (dataSetOriginal) {
    allData = allData.concat(dataSetOriginal);
  }
  if (dataSetSimp) {
    allData = allData.concat(dataSetSimp);
  }

  let rawMax = 1;
  let rawMin = -1;
  if (allData.length > 0) {
    rawMax = Math.max(...allData);
    rawMin = Math.min(...allData);
  }

  const niceMax = Math.ceil(rawMax);
  const niceMin = Math.floor(rawMin);


  const data_label = Array.from({ length: dataSetOriginal.length }, (_, i) => i);
  const state = {
    dataSet: [dataSetCurrent, dataSetSimp, dataSetOriginal],
    labels: data_label,
    options: {
      tooltips: { enabled: true },
      scales: {
        x: [
          {
            gridLines: { display: true, color: "grey" },
            ticks: {
              fontColor: "#3C3C3C",
              fontSize: 14,
              callback: function (value, index) {
                const step_size = 2
                return index % step_size == 0 ? value : null;
              }
            }
          }
        ],
        y: [
          {
            scaleLabel: {
              display: true,
              labelString: "Domain Spesific Y label",
              fontSize: 14
            },
            ticks: {
              display: true,
              suggestedMin: niceMin,
              suggestedMax: niceMax,
              stepSize: 1,
              maxTicksLimit: 10,
              fontColor: "#000000",
              padding: 30,
              callback: function (value, index) {
                const step_size = 10
                return index % step_size === 0 ? value : null;
              }
            },
            gridLines: {
              display: true,
              offsetGridLines: false,
              color: "#3C3C3C",
              tickMarkLength: 4
            }
          }
        ]
      },
      legend: {
        display: true,
        labels: { fontSize: 20, fontStyle: "bold", padding: 16 }
      },
      dragData: true,
      dragOptions: {
        showTooltip: true
      },
      dragDataRound: 1,
      onDragStart: function (e) {
        //console.log("Start:", e);
      },
      onDrag: function (e, datasetIndex, index, value) {
        //console.log("Drag:", datasetIndex, index, value);
      },
      onDragEnd: function (e, datasetIndex, index, value) {
        //console.log("Drag End:", state.dataSet);
        const newDataSet = state.dataSet[0];
        newDataSet[index] = value;
        updateData([...newDataSet], setDataCurrent);

      }.bind(this)
    }
  };

  //console.log("RENDER");
  const data = {
    labels: state.labels,
    datasets: [
      {
        label: "Original Class Label",
        data: state.dataSet[0],
        lineTension: 0,
        borderColor: lineColorCurr,
        borderWidth: 3,
        pointRadius: 2,//7
        pointHoverRadius: 12,
        pointBackgroundColor: "black",
        pointBorderWidth: 0,
        spanGaps: false,
        dragData: true,
        fill: false,
        borderDash: [6, 6]

      },


      {
        label: "Simplification Class Label",
        data: state.dataSet[1],
        lineTension: 0,
        borderColor: lineColorSimp,
        borderWidth: 5,
        pointRadius: 0,
        pointHoverRadius: 1,
        pointBackgroundColor: lineColorSimp,
        pointBorderWidth: 0,
        spanGaps: false,
        dragData: false,
        fill: false,

      }/*,
      {
        label: "Prototype",
        data: state.dataSet[2],
        lineTension: 0,
        borderColor: lineColorOrg,
        borderWidth: 5,
        pointRadius: 1,
        pointHoverRadius: 1,
        pointBackgroundColor: lineColorOrg,
        pointBorderWidth: 0,
        spanGaps: false,
        dragData: false,
        fill: false

      }*/
    ]
  };
  return (
    <div>
      <Line data={data} options={state.options} />
    </div>
  );
};

export default DraggableGraph;
