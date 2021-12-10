using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PointsGUI
{
    public class DllLoader
    {
        const string DllName = "PointsBot.dll";
        private IntPtr handle = IntPtr.Zero;

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "Init")]
        public static extern IntPtr DllInit(int width, int height, int seed);
    }
}
