"use client";

import * as api from '@/utils/api';
import { Icon360View, IconDownload } from '@tabler/icons-react';
              
import { useMemo } from 'react';
import {
  MantineReactTable,
  useMantineReactTable,
  type MRT_ColumnDef,
  type MRT_Cell,
  MRT_ExpandButton,
} from 'mantine-react-table';
import Link from 'next/link';
import { ActionIcon, Box } from '@mantine/core';

function formatDuration(time: number) {
  if (isNaN(time)) return "N/A";
  if (time > 3600) {
    return `${Math.floor(time / 3600)}h ${Math.floor(time / 60) % 60}m ${Math.ceil(time % 60)}s`;
  } else if (time > 60) {
    return `${Math.floor(time / 60)}m ${Math.ceil(time % 60)}s`;
  } else {
    return `${Math.ceil(time)}s`;
  }
}

function formatMemory(memory: number) {
  if (isNaN(memory)) return "N/A";
  if (memory > 1024) {
    return `${(memory / 1024).toFixed(2)} GB`
  } else {
    return `${(memory).toFixed(2)} MB`
  }
}

function formatNumber(number: number) {
  if (isNaN(number)) return "-";
  return number.toFixed(3);
}

export default function DatasetResultsTable({
  datasetId,
  methodResults,
  metrics,
  metricDetail,
  scenes,
  showMethod,
  defaultMetric,
}: {
  datasetId: string,
  methodResults: api.MethodResults[],
  metrics: api.Metric[],
  scenes: api.Scene[],
  metricDetail?: string,
  showMethod?: boolean,
  defaultMetric: string,
}) {
  let enableExpanding = (metricDetail === undefined);
  const sign = metrics.find((m: api.Metric) => m.id === defaultMetric)?.ascending?-1:1;
  const getDefaultMetric = (row: any) => (!row || row[defaultMetric] === undefined || row[defaultMetric] === null) ? Infinity : row[defaultMetric] * sign;
  methodResults = methodResults.sort((a: any, b: any) => getDefaultMetric(a) - getDefaultMetric(b));
  const columns = useMemo<MRT_ColumnDef<any>[]>(
    () => {
      let list: MRT_ColumnDef<any>[] = [];
      if (showMethod ?? true) list.push({
          accessorKey: 'name',
          header: 'Method',
          Cell: ({ cell }: { cell: MRT_Cell<api.MethodResults> }) => (
            <Link href={`/m-${cell.getContext().row.original.id.replace(":", "--")}`}>{cell.getValue() as any}</Link>
          ),
        });
      else list.push({
          accessorKey: 'name',
          header: 'Scene',
          minSize: 30,
          Cell: ({ cell }: { cell: MRT_Cell<api.MethodResults> }) => (
            <>&nbsp;&nbsp;{cell.getValue()}</>
          ),
          AggregatedCell: ({ cell }: { cell: MRT_Cell<api.MethodResults> }) => (
            <>average</>
          ),
        });
      if (metricDetail !== undefined) {
        list = list.concat(scenes.map((scene, i) => ({
          accessorFn: (row) => row.scenes[i] && row.scenes[i][metricDetail],
          header: scene.name ?? scene.id,
          size: 30,
          minSize: 30,
          enableClickToCopy: true,
          Cell: ({ cell }: { cell: MRT_Cell<api.MethodResults> }) => (
            <span>{formatNumber(cell.getValue<number>())}</span>
          )
        })));
      } else {
        list = list.concat(metrics.map((metric) => ({
          accessorKey: `${metric.id}`,
          header: metric.name,
          size: 30,
          minSize: 30,
          enableClickToCopy: true,
          Cell: ({ cell }: { cell: MRT_Cell<api.MethodResults> }) => (
            <span>{formatNumber(cell.getValue<number>())}</span>
          )
        })));
        list.push({
          accessorKey: 'total_train_time',
          header: 'Time',
          size: 30,
          minSize: 30,
          Cell: ({ cell }: { cell: MRT_Cell<api.MethodResults> }) => (
            <span style={{textAlign: "right"}}>{formatDuration(cell.getValue<number>())}</span>
          ),
        });
        list.push({
          accessorKey: 'gpu_memory',
          header: 'GPU Mem.',
          size: 30,
          minSize: 30,
          Cell: ({ cell }: { cell: MRT_Cell<api.MethodResults> }) => (
            <span>{formatMemory(cell.getValue<number>())}</span>
          )
        });
      }
      return list;
    },
    [scenes, metrics]
  );
  const table = useMantineReactTable({
    columns,
    data: methodResults,
    enableColumnActions: false,
    enableColumnFilters: false,
    enablePagination: false,
    enableDensityToggle: false,
    enableSorting: true,
    enableExpanding: enableExpanding,
    enableRowActions: enableExpanding,
    positionExpandColumn: "last",
    displayColumnDefOptions: {
      "mrt-row-expand": {
        header: "",
      },
      "mrt-row-actions": {
        header: "",
      },
    },
    renderRowActions: ({ row }) => 
      row.getCanExpand() ? (
        <MRT_ExpandButton row={row} table={table} />
        ) : (
        <Box sx={{ display: 'flex', flexWrap: 'nowrap', gap: '8px' }}>
          {row.original.demo_link && (<ActionIcon color="cyan"><Link href={ row.original.demo_link}><Icon360View size={26} /></Link></ActionIcon>)}
          {row.original.data_link && (<ActionIcon color="green"><Link href={row.original.data_link}><IconDownload size={26} /></Link></ActionIcon>)}
        </Box>
      ),
    getSubRows: (originalRow: any, index: number) => {
      if (originalRow.scenes !== undefined) {
        let lookup: any = {};
        originalRow.scenes.forEach((m: any) => {lookup[m.id] = m;});
        return scenes.map((scene) => ({
          ...scene,
          ...lookup[scene.id] || {},
        }));
      }
    },
    mantineExpandButtonProps: ({ row, table }) => (row.getCanExpand() ? ({}) : ({disabled:true, style: {visibility: "hidden"}})),
    enableExpandAll: false,
    enableBottomToolbar: false,
    enableTopToolbar: false,
    initialState: {
      density: "xs",
      columnVisibility: {
        'mrt-row-expand': false,
      },
    },
    mantineTableProps: {
      withColumnBorders: false,
      withBorder: false,
    }
  });
  return (
    <MantineReactTable table={table} />
  );
}