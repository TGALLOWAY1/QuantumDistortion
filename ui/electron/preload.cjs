const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openAudioFile: () => ipcRenderer.invoke('open-audio-file'),
  saveAudioFile: (buffer, name) => ipcRenderer.invoke('save-audio-file', buffer, name),
});
