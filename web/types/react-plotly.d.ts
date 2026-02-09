declare module "react-plotly.js" {
  import { Component } from "react";

  interface PlotParams {
    data: Array<Record<string, unknown>>;
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    style?: React.CSSProperties;
    className?: string;
    onInitialized?: (figure: unknown, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: unknown, graphDiv: HTMLElement) => void;
    onClick?: (event: unknown) => void;
  }

  class Plot extends Component<PlotParams> {}
  export default Plot;
}
