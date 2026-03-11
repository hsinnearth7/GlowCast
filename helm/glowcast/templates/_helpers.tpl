{{/*
Expand the name of the chart.
*/}}
{{- define "glowcast.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "glowcast.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "glowcast.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "glowcast.labels" -}}
helm.sh/chart: {{ include "glowcast.chart" . }}
{{ include "glowcast.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "glowcast.selectorLabels" -}}
app.kubernetes.io/name: {{ include "glowcast.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API selector labels.
*/}}
{{- define "glowcast.api.selectorLabels" -}}
{{ include "glowcast.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Dashboard selector labels.
*/}}
{{- define "glowcast.dashboard.selectorLabels" -}}
{{ include "glowcast.selectorLabels" . }}
app.kubernetes.io/component: dashboard
{{- end }}

{{/*
ConfigMap name.
*/}}
{{- define "glowcast.configMapName" -}}
{{ include "glowcast.fullname" . }}-config
{{- end }}

{{/*
Secret name.
*/}}
{{- define "glowcast.secretName" -}}
{{ include "glowcast.fullname" . }}-secrets
{{- end }}
